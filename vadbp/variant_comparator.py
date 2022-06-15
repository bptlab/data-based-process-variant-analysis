import tempfile
from copy import copy
from enum import Enum

import graphviz
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, image as mpimg
from pm4py import format_dataframe
from pm4py.algo.discovery.dfg.adapters.pandas import df_statistics
from pm4py.statistics.attributes.pandas import get as attributes_get
from pm4py.statistics.start_activities.pandas import get as sa_get
from pm4py.statistics.end_activities.pandas import get as ea_get
from pm4py.utils import INDEX_COLUMN
from pm4py.visualization.common import save
from pm4py.objects.conversion.log import converter as log_converter

from vadbp import visualization
from vadbp.plotting import categorical_edge_distribution_graph, continuous_edge_distribution_graph, \
    continuous_node_distribution_graph, categorical_node_distribution_graph
from vadbp.statistics import do_mwu, do_chi_squared, do_wilcoxon
from vadbp.preprocessing import prepare_dataframe_mean_per_activity_measurement, join_prepared_dataframes, \
    prepare_dataframe_mode_per_activity_measurement, prepare_dataframe_edges, prepare_dataframe_edges_categorical, \
    join_prepared_dataframes_on_edges, prepare_dataframe_edges_continuous
from vadbp.visualization import node_mappings_from_gviz, gviz_to_html


class DataType(Enum):
    CONTINUOUS = 0
    CATEGORICAL = 1


def _filter_by_attributes(df, attributes):
    if attributes != "ALL":
        return df[df['variable'].isin(attributes)]
    return df


def _evaluate_statistical_test(test_results, sort_keys, sort_ascending):
    test_results["count_l"] = test_results.apply(lambda x: len(x["value_l"]), axis=1)
    test_results["count_r"] = test_results.apply(lambda x: len(x["value_r"]), axis=1)
    # bonferroni correction
    test_results["p-val-threshold"] = test_results.apply(
        lambda x: 0.05 / max(1, min(len(x["value_l"]), len(x["value_r"]))),
        axis=1)
    test_results = test_results[test_results["p-val"] < test_results["p-val-threshold"]]
    return test_results.sort_values(sort_keys, ascending=sort_ascending)


def get_column_frequency_information(df):
    column_array = []
    for col in df.columns:
        col_factor = df[col].unique().size / df[col].count()
        column_array.append({'column': col, 'unique_count': df[col].unique().size, 'value_count': df[col].count(),
                             'factor': col_factor})
    return pd.DataFrame(column_array).sort_values(['factor'])


class VariantComparator:
    # needs values as expected by pm4py

    def __init__(self, variant1, variant2, original, variant1_name='variant 1', variant2_name='variant 2'):
        self.variant1 = variant1
        self.variant2 = variant2
        self.original = original
        self.variant1_name = variant1_name
        self.variant2_name = variant2_name

        self.column_frequency = get_column_frequency_information(self.original)

        self.activity_comparison_results = {
            DataType.CONTINUOUS: None,
            DataType.CATEGORICAL: None
        }
        self.edge_comparison_results = {
            DataType.CONTINUOUS: None,
            DataType.CATEGORICAL: None
        }
        self.dfg = None
        self.node_id_mapping = {}

    def get_columns_for_type(self, data_type=DataType.CONTINUOUS, threshold=0.04, selected_attributes='ALL'):
        assert isinstance(data_type, DataType), "data type not supported. Use Continuous or Categorical"
        if data_type == DataType.CONTINUOUS:
            columns_for_type = self._get_continuous_columns(threshold)
        else:
            columns_for_type = self._get_categorical_columns(threshold)

        if selected_attributes == 'ALL':
            return self._sort_variables(columns_for_type)
        return self._sort_variables(list(set(columns_for_type) & set(selected_attributes)))

    def _get_categorical_columns(self, threshold):
        return self.column_frequency[self.column_frequency['factor'] <= threshold]['column'].tolist()

    def _get_continuous_columns(self, threshold):
        return self.column_frequency[self.column_frequency['factor'] > threshold]['column'].tolist()

    def _build_dfg(self):
        # TODO: think about which variant to use?!
        if self.dfg is None:
            self.dfg = df_statistics.get_dfg_graph(self.original, measure="frequency")

    def _sort_variables(self, variables):
        return sorted(variables, key=str.casefold)

    def visualize(self, max_edges=20, output='html', attributes='ALL', data_type=DataType.CONTINUOUS,
                  data_type_threshold=0.04):
        attributes = self.get_columns_for_type(data_type, threshold=data_type_threshold, selected_attributes=attributes)

        self._build_dfg()

        activities_RBC = attributes_get.get_attribute_values(self.original, "concept:name")
        for activity in activities_RBC:
            activities_RBC[activity] = self.most_significant_attribute_for_activity(activity,
                                                                                    attributes=attributes,
                                                                                    data_type=data_type)

        edges_RBC_diff = copy(self.dfg)
        for edge in edges_RBC_diff:
            edges_RBC_diff[edge] = self.most_significant_attribute_for_edge(edge,
                                                                            attributes=attributes,
                                                                            data_type=data_type)
        start_activities = sa_get.get_start_activities(self.original)
        end_activities = ea_get.get_end_activities(self.original)

        gviz = visualization.apply(self.dfg, activities_RBC=activities_RBC, edges_RBC=edges_RBC_diff,
                                   parameters={"start_activities": start_activities, "end_activities": end_activities,
                                               "maxNoOfEdgesInDiagram": max_edges})

        self.node_id_mapping = node_mappings_from_gviz(gviz)

        if output == 'plot':
            print(f"used attributes: {attributes}")

            fig = plt.figure(figsize=(15, 8))

            file_name = tempfile.NamedTemporaryFile(suffix='.png')
            file_name.close()

            save.save(gviz, file_name.name)

            img = mpimg.imread(file_name.name)
            plt.axis('off')
            plt.imshow(img)
            plt.show()

        elif output == 'html':
            return gviz_to_html(gviz)

    def _node_id_mapping(self, data_id):
        return self.node_id_mapping.get(data_id, None)

    def dataframe_for_node(self, data_id, attributes='ALL', data_type=DataType.CONTINUOUS,
                           data_type_threshold=0.04):
        attributes = self.get_columns_for_type(data_type, threshold=data_type_threshold, selected_attributes=attributes)

        node_name = self._node_id_mapping(data_id)
        if node_name:
            results_filtered = _filter_by_attributes(self.get_activity_comparison_results(data_type).reset_index(),
                                                     attributes)
            if data_type == DataType.CONTINUOUS:
                return self._readable_column_names(results_filtered[
                    results_filtered['concept:name'] == node_name][
                    ['concept:name', 'variable', 'p-val', 'RBC', 'count_l', 'count_r']].set_index(
                    ['concept:name', 'variable']))
            else:
                return self._readable_column_names(results_filtered[
                    results_filtered['concept:name'] == node_name][
                    ['concept:name', 'variable', 'p-val', 'chi2', 'count_l', 'count_r']].set_index(
                    ['concept:name', 'variable']))
        return "No Data"

    def dataframe_for_edge(self, data_id_left, data_id_right, attributes='ALL', data_type=DataType.CONTINUOUS,
                           data_type_threshold=0.04):
        attributes = self.get_columns_for_type(data_type, threshold=data_type_threshold, selected_attributes=attributes)

        node_name_left = self._node_id_mapping(data_id_left)
        node_name_right = self._node_id_mapping(data_id_right)
        if node_name_right and node_name_left:
            results_filtered = _filter_by_attributes(self.get_edge_comparison_results(data_type).reset_index(),
                                                     attributes)
            if data_type == DataType.CONTINUOUS:
                return self._readable_column_names(results_filtered[
                    (results_filtered['concept:name_l'] == node_name_left) &
                    (results_filtered['concept:name_r'] == node_name_right)][
                    ['concept:name_l', 'concept:name_r', 'variable', 'p-val', 'RBC', 'count_l',
                     'count_r']].set_index(
                    ['concept:name_l', 'concept:name_r', 'variable']))
            else:
                return self._readable_column_names(results_filtered[
                    (results_filtered['concept:name_l'] == node_name_left) &
                    (results_filtered['concept:name_r'] == node_name_right)][
                    ['concept:name_l', 'concept:name_r', 'variable', 'p-val', 'chi2', 'count_l', 'count_r']].set_index(
                    ['concept:name_l', 'concept:name_r', 'variable']))
        return "No Data"

    def plot_for_node_attribute(self, data_id, attribute, data_type=DataType.CONTINUOUS):
        node_name = self._node_id_mapping(data_id)
        if node_name:
            categorical_edges = self.get_activity_comparison_results(data_type).reset_index()
            comparison_attributes = categorical_edges[(categorical_edges['concept:name'] == node_name) & (
                    categorical_edges['variable'] == attribute)]
            if comparison_attributes.size > 0:
                comparison_attributes = comparison_attributes.iloc[0]
                if data_type == DataType.CONTINUOUS:
                    return continuous_node_distribution_graph(comparison_attributes, attribute, self.variant1_name,
                                                              self.variant2_name)
                else:
                    return categorical_node_distribution_graph(comparison_attributes, attribute, self.variant1_name,
                                                               self.variant2_name)
        return "No Data"

    def plot_for_edge_attribute(self, data_id_left, data_id_right, attribute, data_type=DataType.CONTINUOUS):
        node_name_left = self._node_id_mapping(data_id_left)
        node_name_right = self._node_id_mapping(data_id_right)
        if node_name_right and node_name_left:
            categorical_edges = self.get_edge_comparison_results(data_type).reset_index()
            comparison_attributes = categorical_edges[(categorical_edges['concept:name_l'] == node_name_left) & (
                    categorical_edges['concept:name_r'] == node_name_right) & (
                                                              categorical_edges['variable'] == attribute)]
            if comparison_attributes.size > 0:
                comparison_attributes = comparison_attributes.iloc[0]
                if data_type == DataType.CONTINUOUS:
                    return continuous_edge_distribution_graph(comparison_attributes, attribute, self.variant1_name,
                                                              self.variant2_name)
                else:
                    return categorical_edge_distribution_graph(comparison_attributes, attribute, self.variant1_name,
                                                               self.variant2_name)
        return "No Data"

    def get_activity_comparison_results(self, data_type):
        if data_type in self.activity_comparison_results and self.activity_comparison_results[data_type] is not None:
            return self.activity_comparison_results[data_type]
        raise Exception('No results found')

    def _set_activity_comparison_results(self, data_type, results):
        if data_type in self.activity_comparison_results:
            self.activity_comparison_results[data_type] = results
        else:
            raise Exception('Can not set results')

    def get_edge_comparison_results(self, data_type):
        if data_type in self.edge_comparison_results and self.edge_comparison_results[data_type] is not None:
            return self.edge_comparison_results[data_type]
        raise Exception('No results found')

    def _set_edge_comparison_results(self, data_type, results):
        if data_type in self.edge_comparison_results:
            self.edge_comparison_results[data_type] = results
        else:
            raise Exception('Can not set results')

    def do_activity_comparison(self, data_type=DataType.CONTINUOUS):
        assert isinstance(data_type, DataType), "data type not supported. Use Continuous or Categorical"
        if data_type == DataType.CONTINUOUS:
            self._do_activity_comparison_continuous()
        else:
            self._do_activity_comparison_categorical()

    def do_edge_comparison(self, data_type=DataType.CONTINUOUS):
        assert isinstance(data_type, DataType), "data type not supported. Use Continuous or Categorical"

        if data_type == DataType.CONTINUOUS:
            self._do_edge_comparison_continuous()
        else:
            self._do_edge_comparison_categorical()

    def most_significant_attribute_for_activity(self, activity, attributes='ALL', data_type=DataType.CONTINUOUS,
                                                data_type_threshold=0.04):
        attributes = self.get_columns_for_type(data_type, threshold=data_type_threshold, selected_attributes=attributes)

        df_mwu = self.get_activity_comparison_results(data_type).reset_index()
        df_mwu = _filter_by_attributes(df_mwu, attributes)
        top_mwu = df_mwu.groupby("concept:name").head(1)
        res = top_mwu.reset_index().loc[top_mwu.reset_index()["concept:name"] == activity, "res"]
        if len(res.values) > 0:
            return res.values[0]
        else:
            return 0

    def most_significant_attribute_for_edge(self, edge, attributes='ALL', data_type=DataType.CONTINUOUS,
                                            data_type_threshold=0.04):
        attributes = self.get_columns_for_type(data_type, threshold=data_type_threshold, selected_attributes=attributes)

        df_mwu = self.get_edge_comparison_results(data_type).reset_index()
        df_mwu = _filter_by_attributes(df_mwu, attributes)
        top_mwu = df_mwu.groupby(["concept:name_l", "concept:name_r"]).head(1)
        res = top_mwu.reset_index().loc[
            (top_mwu.reset_index()["concept:name_l"] == edge[0]) & (top_mwu.reset_index()["concept:name_r"] == edge[
                1]), "res"]
        if len(res.values) > 0:
            return res.values[0]
        else:
            return 0

    def _do_activity_comparison_continuous(self):
        # run mann-whitney u test
        self._set_activity_comparison_results(DataType.CONTINUOUS,
                                              self._do_statistical_test(
                                                  prepare_func=prepare_dataframe_mean_per_activity_measurement,
                                                  test_func=do_mwu,
                                                  sort_keys=["concept:name", "RBC_abs"], sort_ascending=False))

    def _do_activity_comparison_categorical(self):
        # run chi-squared test
        self._set_activity_comparison_results(DataType.CATEGORICAL,
                                              self._do_statistical_test(
                                                  prepare_func=prepare_dataframe_mode_per_activity_measurement,
                                                  test_func=do_chi_squared,
                                                  sort_keys=["concept:name", "p-val"], sort_ascending=True))

    def _do_edge_comparison_continuous(self):
        # run wilcoxon test
        self._set_edge_comparison_results(DataType.CONTINUOUS,
                                          self._do_statistical_test(prepare_func=prepare_dataframe_edges_continuous,
                                                                    test_func=do_mwu,
                                                                    join_variants=True,
                                                                    join_func=join_prepared_dataframes_on_edges,
                                                                    sort_keys=["concept:name_l", "concept:name_r",
                                                                               "RBC_abs"],
                                                                    sort_ascending=False)
                                          )

    def _do_edge_comparison_categorical(self):
        self._set_edge_comparison_results(DataType.CATEGORICAL,
                                          self._do_statistical_test(prepare_func=prepare_dataframe_edges_categorical,
                                                                    test_func=do_chi_squared,
                                                                    join_variants=True,
                                                                    join_func=join_prepared_dataframes_on_edges,
                                                                    sort_keys=["concept:name_l", "concept:name_r",
                                                                               "p-val"],
                                                                    sort_ascending=False)
                                          )

    def _do_statistical_test(self, prepare_func, test_func, sort_keys, sort_ascending, join_variants=True,
                             join_func=join_prepared_dataframes):
        variant1_prepared = prepare_func(self.variant1)
        variant2_prepared = prepare_func(self.variant2)
        if join_variants:
            variants_joined = join_func(variant1_prepared, variant2_prepared)
            test_results = variants_joined.apply(test_func, axis=1)
            return _evaluate_statistical_test(test_results, sort_keys, sort_ascending)
        else:
            test_results = []
            for var in [variant1_prepared, variant2_prepared]:
                test_results.append(
                    _evaluate_statistical_test(var.apply(test_func, axis=1), sort_keys, sort_ascending))
            return test_results

    def _readable_column_names(self, df):
        col_names = {
            'count_l': f'Count {self.variant1_name}',
            'count_r': f'Count {self.variant2_name}',
            'p-val_1': f'P-value {self.variant1_name}',
            'p-val_2': f'P-value {self.variant2_name}',
            'RBC_1': f'RBC {self.variant1_name}',
            'RBC_2': f'RBC {self.variant2_name}',
            'count_1': f'Count {self.variant1_name}',
            'count_2': f'Count {self.variant2_name}',
            'concept:name_l': f'',
            'concept:name_r': f'',
            'variable': f'Measurement',
            'p-val': f'P-value',
        }
        return df.rename(columns=col_names)

    def prepare(self):
        # continuous vars
        self.do_activity_comparison(DataType.CONTINUOUS)
        self.do_edge_comparison(DataType.CONTINUOUS)
        # categorical vars
        self.do_activity_comparison(DataType.CATEGORICAL)
        self.do_edge_comparison(DataType.CATEGORICAL)

    @staticmethod
    def format_df(df, case_id, activity_key, timestamp_key):
        return format_dataframe(df, case_id=case_id, activity_key=activity_key, timestamp_key=timestamp_key) \
            .set_index(INDEX_COLUMN)
