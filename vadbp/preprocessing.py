import pandas as pd
import scipy


def prepare_dataframe_edges(variant):
    # variant must be sorted by ['case:concept:name', 'time:timestamp']

    # joined = variant.join(variant.shift(-1).convert_dtypes(convert_string=False, convert_integer=False, convert_boolean=False, convert_floating=True), lsuffix='_l', rsuffix='_r')
    joined = variant.join(variant.shift(-1), lsuffix='_l', rsuffix='_r')
    joined = joined[joined['case:concept:name_l'] == joined['case:concept:name_r']]

    df_aggregated = joined.groupby(['concept:name_l', 'concept:name_r']).agg(lambda x: x.tolist())
    df_melted = pd.melt(df_aggregated.reset_index(), id_vars=['concept:name_l', 'concept:name_r'],
                        value_vars=df_aggregated.columns)
    df_final = df_melted[df_melted['value'].map(lambda d: len(d)) > 0]

    left_side_df = df_final[df_final['variable'].str.endswith('_l')]
    left_side_df['variable'] = left_side_df['variable'].apply(lambda row: row.rstrip('_l'))
    right_side_df = df_final[df_final['variable'].str.endswith('_r')]
    right_side_df['variable'] = right_side_df['variable'].apply(lambda row: row.rstrip('_r'))
    merged = left_side_df.merge(right_side_df, on=['variable', 'concept:name_l', 'concept:name_r'],
                                suffixes=['_l', '_r'])

    merged = merged[merged['value_l'] != merged['value_r']]
    return merged.set_index(['concept:name_l', 'concept:name_r', 'variable'])


def prepare_dataframe_edges_categorical(variant):
    variant = prepare_dataframe_edges(variant)
    variant['value'] = variant.apply(
        lambda row: [str(row['value_l'][i]) + '-' + str(row['value_r'][i]) for i in range(0, len(row['value_l']))
                     if (not pd.isna(row['value_l'][i]) and not pd.isna(row['value_r'][i]))],
        axis=1)
    variant.drop(['value_l', 'value_r'], axis=1, inplace=True)
    return variant


def prepare_dataframe_edges_continuous(variant):
    variant = prepare_dataframe_edges(variant)
    variant['value'] = variant.apply(
        lambda row: [float(row['value_r'][i]) - float(row['value_l'][i]) for i in range(0, len(row['value_l']))
                     if (pd.api.types.is_numeric_dtype(type(row['value_l'][i])) and pd.api.types.is_numeric_dtype(type(row['value_r'][i])) and not pd.isna(row['value_l'][i]) and not pd.isna(row['value_r'][i]))],
        axis=1)
    variant.drop(['value_l', 'value_r'], axis=1, inplace=True)
    return variant


def prepare_dataframe_mean_per_activity_measurement(df):
    df_meaned = df.groupby(["case:concept:name", "concept:name"]).mean()
    return melt_prepared_dataframe(df_meaned)


def prepare_dataframe_mode_per_activity_measurement(df):
    df_moded = df.groupby(["case:concept:name", "concept:name"]).agg(
        lambda x: scipy.stats.mode(x)[0])
    return melt_prepared_dataframe(df_moded)


def melt_prepared_dataframe(df_prepared):
    df_aggregated = df_prepared.groupby(['concept:name']).agg(lambda x: x.dropna().tolist())
    df_melted = pd.melt(df_aggregated.reset_index(), id_vars='concept:name',
                        value_vars=df_aggregated.columns
                        )
    df_final = df_melted[df_melted['value'].map(lambda d: len(d)) > 0]
    return df_final.set_index(['concept:name', 'variable'])


def join_prepared_dataframes(d1, d2):
    return d1.join(d2, on=['concept:name', 'variable'], how='inner', lsuffix='_l', rsuffix='_r')


def join_prepared_dataframes_on_edges(d1, d2):
    return d1.join(d2, on=['concept:name_l', 'concept:name_r', 'variable'], how='inner', lsuffix='_l', rsuffix='_r')
