#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2022/1/30
Time: 20:17
File: 2_0_dynamic_threshold.py
HomePage : https://github.com/meiyuanqing
Email : dg1533019@smail.nju.edu.cn

dynamic_threshold:
    Apply the confounding removal model to achieve the dynamic threshold.
        The confounding removal model:
        X = a + bZ
        X: a OO metric
        Z: size metric (SLOC)

"""


def bender_auc_threshold(df):
    import statsmodels.api as sm
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix

    metric_name = ''
    for col in df.columns:
        if col not in ['bug', 'bugBinary', 'intercept']:
            metric_name = col

    try:
        rus = RandomOverSampler(random_state=0)
        # rus = RandomUnderSampler(random_state=0)
        X_resampled, y_resampled = rus.fit_resample(df.loc[:, [metric_name, 'intercept']], df['bugBinary'])
        # print(y_resampled)
        # print(X_resampled)
        logit = sm.Logit(y_resampled, X_resampled)  # RandomUnderSampler

        # logit = sm.Logit(df['bugBinary'], df.loc[:, [metric_name, 'intercept']])
        result_logit = logit.fit()
        # print(result_logit.summary())
    except Exception as err1:
        print(err1)
        return 0, 0, 0, 0

    tau = result_logit.params[0]
    beta = result_logit.params[1]
    tau_pvalue = result_logit.pvalues[0]
    beta_removed_pvalue = result_logit.pvalues[1]

    # 1. bender method to derive threshold for metric_inverted_comma
    if tau == 0:
        bender_t = 0
    else:
        valueOfbugBinary = df['bugBinary'].value_counts()  # 0 和 1 的各自的个数

        # 用缺陷为大于0的模块数占所有模块之比
        BaseProbability_1 = valueOfbugBinary[1] / (valueOfbugBinary[0] + valueOfbugBinary[1])
        # 计算VARL阈值
        bender_t = (np.log(BaseProbability_1 / (1 - BaseProbability_1)) - beta) / tau

    # 判断每个度量与bug之间的关系,用于阈值判断正反例
    Corr_metric_bug = df.loc[:, [metric_name, 'bug']].corr('spearman')

    Spearman_value = Corr_metric_bug[metric_name][1]
    Pearson_value = 2 * np.sin(np.pi * Spearman_value / 6)

    return Pearson_value, tau, tau_pvalue, bender_t


def dynamic_threshold(working_dir, result_dir):
    import statsmodels.api as sm

    # display all columns and rows, and set the item of row of dataframe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    working_directory = working_dir
    result_directory = result_dir
    os.chdir(working_directory)

    with open(working_directory + "List.txt") as l_all:
        lines_all = l_all.readlines()

    for line in lines_all:

        project = line.replace("\n", "")
        print("the file is ", project)
        if project != 'groovy':
            continue

        dynamicThreshold = pd.DataFrame(
            columns=['version', 'metric', 'pearson_removed', 'tau_removed', 'tau_removed_pvalue', 'bender_t', 'a_const',
                     'b_SLOC', 'a_const_pvalue', 'b_SLOC_pvalue', 'a_const_variance', 'b_SLOC_variance', 't_max',
                     't_min', 'bender_t0', 'sloc_max', 'sloc_min', 'sloc_median', 'sloc_avg', 'sloc_var'])

        for root, dirs, files in os.walk(working_directory + project):
            for name in files:

                df_name = pd.read_csv(working_directory + project + '\\' + name)

                # exclude the non metric fields and 31 size metrics
                non_metric = ["relName", "className", 'Kind', 'Name', 'File', "bug", 'prevsloc', 'currsloc',
                              'addedsloc', 'deletedsloc', 'changedsloc', 'totalChangedsloc', 'SLOC', 'NA', 'NAIMP',
                              'NCM', 'NIM', 'NLM', 'NM', 'NMIMP', 'NMNpub', 'NMpub', 'NTM', 'NumPara', 'stms']

                # metric_data stores the metric fields (102 items)
                def fun_1(m):
                    return m if m not in non_metric else None

                metric_data = filter(fun_1, df_name.columns)

                for metric in metric_data:

                    print("the current file is ", name, "the current metric is ", metric)

                    # 由于bug中存储的是缺陷个数,转化为二进制存储,若x>2,则可预测bug为3个以上的阈值,其他类推
                    df_name['bugBinary'] = df_name.bug.apply(lambda x: 1 if x > 0 else 0)

                    # 删除度量中空值和undef值
                    df_metric = df_name[~df_name[metric].isin(['undef', 'undefined'])].loc[:, ['bug', 'bugBinary',
                                                                                               metric, 'SLOC']]

                    df_metric = df_metric.dropna(subset=[metric]).reset_index(drop=True)

                    metric_min = df_metric[metric].astype(float).min()
                    metric_max = df_metric[metric].astype(float).max()

                    sloc_max = df_metric['SLOC'].astype(float).max()
                    sloc_min = df_metric['SLOC'].astype(float).min()
                    sloc_median = df_metric['SLOC'].astype(float).median()
                    sloc_avg = df_metric['SLOC'].astype(float).mean()
                    sloc_var = df_metric['SLOC'].astype(float).var()

                    # exclude those data sets in which the metric m has fewer than six non-zero data points
                    # (each corresponding to a class).
                    if len(df_metric) - len(df_metric[df_metric[metric] == 0]) < 6:
                        continue

                    df_metric['const'] = 1.0
                    Z_SLOC = df_metric['SLOC'].astype(float)
                    X_metric = df_metric[metric].astype(float)
                    Z_SLOC = sm.add_constant(Z_SLOC.values)

                    # print(type(X_metric))
                    # print(type(Z_SLOC))
                    # est = sm.OLS(X_metric.astype(float), Z_SLOC.astype(float))
                    est = sm.OLS(df_metric[metric].astype(float), df_metric.loc[:, ['const', 'SLOC']].astype(float))
                    est = est.fit()
                    # print(est.summary())
                    a_const_pvalue = est.pvalues[0]
                    b_SLOC_pvalue = est.pvalues[1]
                    a_const = est.params[0]
                    b_SLOC = est.params[1]
                    # print(a_const, b_SLOC, a_const_pvalue, b_SLOC_pvalue)
                    a_const_95CI_LL = est.conf_int().loc['const', 0]
                    a_const_95CI_UL = est.conf_int().loc['const', 1]
                    b_SLOC_95CI_LL = est.conf_int().loc['SLOC', 0]
                    b_SLOC_95CI_UL = est.conf_int().loc['SLOC', 1]
                    # print(a_const_95CI_LL, a_const_95CI_UL, b_SLOC_95CI_LL, b_SLOC_95CI_UL)
                    a_const_variance = est.cov_params().loc['const', 'const']
                    b_SLOC_variance = est.cov_params().loc['SLOC', 'SLOC']
                    a_const_std = a_const_variance ** 0.5
                    b_SLOC_std = b_SLOC_variance ** 0.5
                    # print(a_const_variance, b_SLOC_variance, a_const_std, b_SLOC_std)
                    X_metric_hat = est.predict(Z_SLOC)
                    # print(X_metric_hat.head())

                    # logit model for removed effect size and derive five kinds of threshold
                    df_metric['metric_inverted_comma'] = X_metric.astype(float) - X_metric_hat.astype(float)
                    df_metric['intercept'] = 1.0

                    pearson_t, tau_t, tau_p_t, bender_t = bender_auc_threshold(
                        df_metric.loc[:, ['bug', 'bugBinary', 'metric_inverted_comma', 'intercept']].astype(float))

                    print(pearson_t, tau_t, tau_p_t, bender_t)
                    # if pearson correlation is zero, continue
                    if pearson_t == 0:
                        continue

                    dynamicThreshold = dynamicThreshold.append(
                        {'version': name[:-4], 'metric': metric, 'pearson_removed': pearson_t, 'tau_removed': tau_t,
                         'tau_removed_pvalue': tau_p_t, 'bender_t': bender_t, 'a_const': a_const,
                         'b_SLOC': b_SLOC, 'a_const_pvalue': a_const_pvalue, 'b_SLOC_pvalue': b_SLOC_pvalue,
                         'a_const_variance': a_const_variance, 'b_SLOC_variance': b_SLOC_variance,
                         't_max': metric_max, 't_min': metric_min, 'bender_t0': (bender_t + a_const + b_SLOC),
                         'sloc_max': sloc_max, 'sloc_min': sloc_min, 'sloc_median': sloc_median, 'sloc_avg': sloc_avg,
                         'sloc_var': sloc_var
                         }, ignore_index=True)
                    # break
                dynamicThreshold.to_csv(result_directory + 'dynamicThreshold_' + project + '.csv', index=False)
                # break
        # break


if __name__ == '__main__':
    import os
    import sys
    import csv
    import math
    import time
    import random
    import shutil
    from datetime import datetime
    import pandas as pd
    import numpy as np

    s_time = time.time()

    working_Directory = "F:\\DT\\pyData\\data_defects_java_training\\"
    result_Directory = "F:\\DT\\dynamicThresholds\\"
    spv_result_Directory = "F:\\DT\\dynamicThresholds_SPV\\"
    os.chdir(working_Directory)

    # 1. deriving dynamic thresholds of OO metrics on each release
    dynamic_threshold(working_Directory, result_Directory)

    # 2. predict the defect-prone class on the consecutive release,i.e.,using the former threshold.
    # dynamic_threshold_spv(working_Directory, spv_result_Directory)

    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ".\nFrom ", time.asctime(time.localtime(s_time)), " to ",
          time.asctime(time.localtime(e_time)), ",\nThis", os.path.basename(sys.argv[0]), "ended within",
          execution_time, "(s), or ", (execution_time / 60), " (m).")
