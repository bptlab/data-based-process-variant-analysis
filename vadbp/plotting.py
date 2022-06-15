import tempfile

import graphviz
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, image as mpimg
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

VARIANT_1_COLOR = '#1f77b4'
VARIANT_2_COLOR = '#ff7f0e'


def categorical_edge_distribution_graph(comparison_attributes, comparison_attribute, variant1_name, variant2_name):
    df_l = pd.DataFrame(comparison_attributes['value_l'], columns=['edge'])
    df_r = pd.DataFrame(comparison_attributes['value_r'], columns=['edge'])
    df_l = df_l.groupby('edge').size().reset_index(name='freq_l').set_index('edge')
    df_r = df_r.groupby('edge').size().reset_index(name='freq_r').set_index('edge')

    merged = df_l.join(df_r, how='outer').fillna(0).reset_index()
    merged[['source', 'dest']] = merged['edge'].str.split('-', expand=True, n=1)
    merged = merged.convert_dtypes()

    # create graph and nodes
    dot = graphviz.Digraph()
    dot.graph_attr['rankdir'] = 'LR'
    nodes = np.unique(np.concatenate((merged['source'], merged['dest']), axis=0))
    for node_name in nodes:
        dot.node(node_name, node_name)

    # add edges
    for index, row in merged.iterrows():
        frequency_left = round(row['freq_l'] / comparison_attributes['count_l'], 2)
        frequency_right = round(row['freq_r'] / comparison_attributes['count_r'], 2)
        label = f"""<
            <font color='{VARIANT_1_COLOR}'> {(100*frequency_left):2.1f} % </font>  /  <font color='{VARIANT_2_COLOR}'> {(100*frequency_right):2.1f} % </font>
            >"""
        dot.edge(row['source'], row['dest'], label=label)

    # plot in matplotlib to add legend
    file_name = tempfile.NamedTemporaryFile(suffix='.png')
    file_name.close()
    dot.render(filename=file_name.name, format='png')
    img = mpimg.imread(file_name.name + ".png")

    fig = plt.figure(figsize=(15, 5))
    plt.imshow(img)
    plt.axis('off')
    legend_elements = [Patch(label='over 60', color=VARIANT_1_COLOR),
                       Patch(label='under 60', color=VARIANT_2_COLOR)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0)
    return fig


def continuous_edge_distribution_graph(comparison_attributes, comparison_attribute, variant1_name, variant2_name):
    diff_left = np.array(comparison_attributes['value_l'])
    diff_right = np.array(comparison_attributes['value_r'])
    fig = plt.figure(figsize=(15, 5))
    plt.hist(diff_left, histtype='step', label=variant1_name, alpha=1, linewidth=2.5, density=True,
             color=VARIANT_1_COLOR)
    plt.hist(diff_right, histtype='step', label=variant2_name, alpha=1, linewidth=2.5, density=True,
             color=VARIANT_2_COLOR)
    plt.legend(loc='best')
    plt.axvline(x=0)
    plt.xlabel(f'{comparison_attribute} change in edge')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.ylabel('%')
    return fig


def categorical_node_distribution_graph(comparison_attributes, comparison_attribute, variant1_name, variant2_name):
    fig, ax = plt.subplots(figsize=(15, 5))
    all_labels = np.unique(np.concatenate((comparison_attributes['value_l'], comparison_attributes['value_r'])))
    labels_left = np.concatenate((comparison_attributes['value_l'], all_labels))
    labels_right = np.concatenate((comparison_attributes['value_r'], all_labels))
    labels_left.sort()
    labels_right.sort()
    labels_left, counts_left = np.unique(labels_left, return_counts=True)
    labels_right, counts_right = np.unique(labels_right, return_counts=True)
    counts_left = (counts_left - 1) / len(comparison_attributes['value_l'])
    counts_right = (counts_right - 1) / len(comparison_attributes['value_r'])

    ind = np.arange(all_labels.size)
    width = 0.2

    ax.bar(ind, counts_left, width, label=variant1_name, color=VARIANT_1_COLOR)
    ax.bar(ind + width, counts_right, width, label=variant2_name, color=VARIANT_2_COLOR)
    ax.set_xticks(ind + width / 2, rotation=90)
    ax.set_xticklabels(all_labels)
    ax.legend(loc='best')
    ax.set_xlabel(comparison_attribute)
    ax.set_ylabel('%')
    plt.xticks(rotation=90)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    return fig


def continuous_node_distribution_graph(comparison_attributes, comparison_attribute, variant1_name, variant2_name):
    fig = plt.figure(figsize=(15, 5))
    plt.hist(comparison_attributes['value_l'], label=variant1_name, histtype='step', alpha=1, linewidth=2.5,
             density=True, color=VARIANT_1_COLOR)
    plt.hist(comparison_attributes['value_r'], label=variant2_name, histtype='step', alpha=1, linewidth=2.5,
             density=True, color=VARIANT_2_COLOR)
    plt.legend(loc='best')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel(comparison_attribute)
    plt.ylabel('%')
    return fig
