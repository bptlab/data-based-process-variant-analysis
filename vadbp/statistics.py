import numpy as np
import pandas as pd
import pingouin as pg


def do_mwu(row):
    if len(row["value_l"]) > 1 and len(row["value_r"]) > 1:
        mwu_results = pg.mwu(row["value_l"], row["value_r"])
        row['p-val'] = mwu_results['p-val'][0]
        row['RBC'] = mwu_results['RBC'][0]
        row['RBC_abs'] = abs(mwu_results['RBC'][0])
        row['CLES'] = mwu_results['CLES'][0]

        row['res'] = row['RBC_abs']
        return row
    return row


def do_wilcoxon(row):
    try:
        if row["value_l"] != row["value_r"]:
            mwu_results = pg.wilcoxon(row["value_l"], row["value_r"])
            row['p-val'] = mwu_results['p-val'][0]
            row['RBC'] = mwu_results['RBC'][0]
            row['RBC_abs'] = abs(mwu_results['RBC'][0])
            row['CLES'] = mwu_results['CLES'][0]

            row['res'] = row['RBC_abs']
            return row
        else:
            # links = rechts
            return row
    except Exception:
        return row


def do_chi_squared(row):
    left_df = pd.DataFrame(row["value_l"], columns=['values'])
    left_df['category'] = 'left'
    right_df = pd.DataFrame(row["value_r"], columns=['values'])
    right_df['category'] = 'right'
    df_concat = pd.concat([left_df, right_df])
    try:
        _, _, stats = pg.chi2_independence(df_concat, x='category', y='values')
        pearson_stats = stats[stats['test'] == 'pearson']
    except:
        return row
    row['p-val'] = pearson_stats['pval'][0]
    row['chi2'] = pearson_stats['chi2'][0]
    row['dof'] = pearson_stats['dof'][0]
    # pval und chi2 value for comparison
    row['power'] = pearson_stats['power'][0]
    row['cramer'] = pearson_stats['cramer'][0]

    row['res'] = pearson_stats['chi2'][0]
    return row
