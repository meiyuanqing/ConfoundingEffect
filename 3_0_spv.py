#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2022/1/30
Time: 22:37
File: 3_0_spv.py
HomePage : https://github.com/meiyuanqing
Email : dg1533019@smail.nju.edu.cn

For dynamic and static threshold of each OO metric:

    SPV uses project data from the latest prior release of a target release for training.
    SPV (Single Prior Version) Project data of the (accessible) latest prior release is used for that purpose.

Reference:
[1] Amasaki, S. Cross-version defect prediction: use historical data, cross-project data, or both?
 Empirical Software Engineering, 25, 2 (2020), 1573-1595.

"""


def static_threshold_spv(working_dir, result_dir):
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix

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
        if project != 'groovy':
            continue
        # if project != 'camel_rm_test':
        #     continue
        print("the file is ", project)

        if os.path.exists(result_directory + 'staticPerformance_' + project + '.csv'):
            print("the csv file is created in last execution, so it will not be created this time.")
            continue

        # df_static_threshold = pd.read_csv('F:\\DT\\staticThresholds\\' + project + '\\staticThreshold_' + project + '.csv')
        # print(df_static_threshold.head())
        df_version_date = pd.read_csv('F:\\MTmeta\\pyData\\versionDate\\' + project.upper() + '_versionDate.csv')
        # print(df_version_date.head())
        # print(df_version_date.name.values)
        versions_sorted = df_version_date.name.values
        print(versions_sorted)
        print(len(versions_sorted))

        staticPerformance = pd.DataFrame(columns=['version', 'metric', 'ER', 'auc_value', 'recall_value',
                                                  'precision_value', 'f1_value', 'gm_value', 'bpp_value', 'bender_t'])

        for root, dirs, files in os.walk(working_directory + project):

            files_sorted = []

            for version in versions_sorted:
                # print(version)
                for file in files:
                    # print(file, file.split('-')[1][:-4], version, file.split('-')[1][:-4] == version)
                    if file.split('-')[1][:-4] == version:
                        files_sorted.append(file)
            print(files_sorted)
            print(len(files_sorted))

            for i in range(len(files_sorted)):
                if i == 0:
                    continue
                print(files_sorted[i - 1])
                df_versoion_threshold = pd.read_csv(
                    'F:\\DT\\staticThresholds\\' + project + '\\staticThreshold_' + files_sorted[i - 1])
                # df_versoion_threshold = df_threshold[df_threshold['version'] == files_sorted[i - 1][:-4]]
                # print(df_versoion_threshold.head())

                for m in df_versoion_threshold['metric'].unique():

                    if m == 'SLOC':
                        continue
                    print("the threshold file is ", files_sorted[i - 1], " the testing is in ", files_sorted[i],
                          " the metric is ", m)
                    df_m = df_versoion_threshold[df_versoion_threshold['metric'] == m].loc[:, :].reset_index()
                    # print(df_m.head())
                    if len(df_m) < 1:
                        continue
                    # if df_m.loc[0, 'tau_pvalue'] > 0.05:
                    #     continue

                    tau = df_m.loc[0, 'tau']
                    pearson = df_m.loc[0, 'pearson']
                    bender_t = df_m.loc[0, 'bender_t']
                    auc_t = df_m.loc[0, 'auc_t']
                    # bender_t0 = df_m.loc[0, 'bender_t0']
                    # a_const = df_m.loc[0, 'a_const']
                    # b_SLOC = df_m.loc[0, 'b_SLOC']
                    # print(tau_removed, bender_t, a_const, b_SLOC)

                    df_versoion_testing = pd.read_csv(working_directory + project + '\\' + files_sorted[i])

                    # print(m, df_versoion_testing.columns.values, m not in df_versoion_testing.columns.values)
                    if m not in df_versoion_testing.columns.values:
                        continue

                    # 由于bug中存储的是缺陷个数,转化为二进制存储,若x>2,则可预测bug为3个以上的阈值,其他类推
                    df_versoion_testing['bugBinary'] = df_versoion_testing.bug.apply(lambda x: 1 if x > 0 else 0)

                    # 删除度量中空值和undef值
                    df_versoion_testing = df_versoion_testing[~df_versoion_testing[m].isin(['undef', 'undefined'])].loc[
                                          :, ['bug', 'bugBinary', m, 'SLOC']]

                    df_versoion_testing = df_versoion_testing.dropna(subset=[m, 'SLOC']).reset_index(drop=True)

                    # 如果bugBinary中只有一个值，即全为零或全为1，mrm-1.1.csv中全为1，全为零在预测处理过滤了。
                    if len(df_versoion_testing['bugBinary'].value_counts()) == 1:
                        continue

                    s_p, s, f_p, f = 0, 0, 0, 0

                    if pearson < 0:
                        for j in range(len(df_versoion_testing)):
                            if float(df_versoion_testing.loc[j, m]) <= auc_t:
                                df_versoion_testing.loc[j, 'predictBinary'] = 1
                                s += df_versoion_testing.loc[j, 'SLOC']
                                s_p += df_versoion_testing.loc[j, 'SLOC'] * 1
                                f += df_versoion_testing.loc[j, 'bug']
                                f_p += df_versoion_testing.loc[j, 'bug'] * 1
                            else:
                                df_versoion_testing.loc[j, 'predictBinary'] = 0
                                s += df_versoion_testing.loc[j, 'SLOC']
                                s_p += df_versoion_testing.loc[j, 'SLOC'] * 0
                                f += df_versoion_testing.loc[j, 'bug']
                                f_p += df_versoion_testing.loc[j, 'bug'] * 0
                    else:
                        for j in range(len(df_versoion_testing)):
                            if float(df_versoion_testing.loc[j, m]) >= auc_t:
                                df_versoion_testing.loc[j, 'predictBinary'] = 1
                                s += df_versoion_testing.loc[j, 'SLOC']
                                s_p += df_versoion_testing.loc[j, 'SLOC'] * 1
                                f += df_versoion_testing.loc[j, 'bug']
                                f_p += df_versoion_testing.loc[j, 'bug'] * 1
                            else:
                                df_versoion_testing.loc[j, 'predictBinary'] = 0
                                s += df_versoion_testing.loc[j, 'SLOC']
                                s_p += df_versoion_testing.loc[j, 'SLOC'] * 0
                                f += df_versoion_testing.loc[j, 'bug']
                                f_p += df_versoion_testing.loc[j, 'bug'] * 0

                    if f == 0:
                        effort_random == 0
                    else:
                        effort_random = f_p / f
                    effort_random = f_p / f
                    effort_m = s_p / s
                    if effort_random == 0:
                        ER_m = 0
                    else:
                        ER_m = (effort_random - effort_m) / effort_random

                    # confusion_matrix()函数中需要给出label, 0和1，否则该函数算不出TP,因为不知道哪个标签是poistive.
                    c_matrix = confusion_matrix(df_versoion_testing["bugBinary"], df_versoion_testing['predictBinary'],
                                                labels=[0, 1])
                    tn, fp, fn, tp = c_matrix.ravel()

                    if (tn + fp) == 0:
                        tnr_value = 0
                    else:
                        tnr_value = tn / (tn + fp)

                    if (fp + tn) == 0:
                        fpr = 0
                    else:
                        fpr = fp / (fp + tn)

                    auc_value = roc_auc_score(df_versoion_testing['bugBinary'], df_versoion_testing['predictBinary'])
                    recall_value = recall_score(df_versoion_testing['bugBinary'], df_versoion_testing['predictBinary'],
                                                labels=[0, 1])
                    precision_value = precision_score(df_versoion_testing['bugBinary'],
                                                      df_versoion_testing['predictBinary'], labels=[0, 1])
                    f1_value = f1_score(df_versoion_testing['bugBinary'], df_versoion_testing['predictBinary'],
                                        labels=[0, 1])

                    gm_value = (recall_value * tnr_value) ** 0.5
                    pfr = recall_value
                    pdr = fpr  # fp / (fp + tn)
                    bpp_value = 1 - (((0 - pfr) ** 2 + (1 - pdr) ** 2) * 0.5) ** 0.5

                    # print(auc_value, recall_value, precision_value, f1_value, gm_value, bpp_value)
                    staticPerformance = staticPerformance.append(
                        {'version': files[i], 'metric': m, 'ER': ER_m, 'auc_value': auc_value,
                         'recall_value': recall_value, 'precision_value': precision_value, 'f1_value': f1_value,
                         'gm_value': gm_value, 'bpp_value': bpp_value, 'bender_t': bender_t}, ignore_index=True)

                    staticPerformance.to_csv(
                        result_directory + 'maxAUCthreshold\\staticPerformance_' + project + '.csv', index=False)
                    # break
                # break
        # break


def dynamic_threshold_spv(working_dir, result_dir):
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc

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
        # if project != 'mrm':
        #     continue
        if project != 'groovy':
            continue
        print("the file is ", project)

        if os.path.exists(result_directory + 'dynamicPerformance_' + project + '.csv'):
            print("the csv file is created in last execution, so it will not be created this time.")
            continue

        df_threshold = pd.read_csv('F:\\DT\\dynamicThresholds\\dynamicThreshold_' + project + '.csv')
        # print(df_threshold.head())
        df_version_date = pd.read_csv('F:\\DT\\pyData\\versionDate\\' + project.upper() + '_versionDate.csv')
        # print(df_version_date.head())
        # print(df_version_date.name.values)
        versions_sorted = df_version_date.name.values
        print(versions_sorted)
        print(len(versions_sorted))

        dynamicPerformance = pd.DataFrame(columns=['version', 'metric', 'ER', 'auc_value', 'recall_value',
                                                   'precision_value', 'f1_value', 'gm_value', 'bpp_value',
                                                   'auc_t', 'a_const', 'b_SLOC'])

        for root, dirs, files in os.walk(working_directory + project):

            files_sorted = []

            for version in versions_sorted:
                # print(version)
                for file in files:
                    # print(file, file.split('-')[1][:-4], version, file.split('-')[1][:-4] == version)
                    if file.split('-')[1][:-4] == version:
                        files_sorted.append(file)
            print(files_sorted)

            for i in range(len(files_sorted)):
                if i == 0:
                    continue
                df_versoion_threshold = df_threshold[df_threshold['version'] == files_sorted[i - 1][:-4]]
                # print(df_versoion_threshold.head())

                # 读入静态阈值
                df_versoion_threshold_static = pd.read_csv(
                    'F:\\DT\\staticThresholds\\' + project + '\\staticThreshold_' + files_sorted[i - 1])

                for m in df_versoion_threshold['metric'].unique():
                    print("the threshold file is ", files_sorted[i - 1], " the testing is in ", files_sorted[i],
                          " the metric is ", m)
                    df_m = df_versoion_threshold[df_versoion_threshold['metric'] == m].loc[:, :].reset_index()
                    # print(df_m.head())
                    if len(df_m) < 1:
                        continue

                    # 过滤掉不能用单个度量（去掉混合效应后）预测的度量
                    if df_m.loc[0, 'tau_removed_pvalue'] > 0.05:
                        continue

                    tau_removed = df_m.loc[0, 'tau_removed']
                    bender_t = df_m.loc[0, 'bender_t']
                    bender_t0 = df_m.loc[0, 'bender_t0']
                    a_const = df_m.loc[0, 'a_const']
                    b_SLOC = df_m.loc[0, 'b_SLOC']

                    # 度量取值的上限和下限
                    t_max = df_m.loc[0, 't_max']
                    t_min = df_m.loc[0, 't_min']

                    df_m_static = df_versoion_threshold_static[
                                      df_versoion_threshold_static['metric'] == m].loc[:, :].reset_index()

                    # 过滤掉没有静态阈值的度量
                    if len(df_m_static) < 1:
                        continue
                    auc_t = df_m_static.loc[0, 'auc_t']
                    # print(tau_removed, bender_t, a_const, b_SLOC)

                    df_versoion_testing = pd.read_csv(working_directory + project + '\\' + files_sorted[i])

                    # print(m, df_versoion_testing.columns.values, m not in df_versoion_testing.columns.values)
                    if m not in df_versoion_testing.columns.values:
                        continue

                    # 由于bug中存储的是缺陷个数,转化为二进制存储,若x>2,则可预测bug为3个以上的阈值,其他类推
                    df_versoion_testing['bugBinary'] = df_versoion_testing.bug.apply(lambda x: 1 if x > 0 else 0)

                    # 删除度量中空值和undef值
                    df_versoion_testing = df_versoion_testing[~df_versoion_testing[m].isin(['undef', 'undefined', 'NaN',
                                          ''])].loc[:, ['bug', 'bugBinary', m, 'SLOC']]

                    df_versoion_testing = df_versoion_testing.dropna(subset=[m, 'SLOC']).reset_index(drop=True)

                    # 如果bugBinary中只有一个值，即全为零或全为1，mrm-1.1.csv中全为1，全为零在预测处理过滤了。
                    if len(df_versoion_testing['bugBinary'].value_counts()) == 1:
                        continue

                    # 判断每个度量与bug之间的关系 .astype(float) 有个别度量是字符型
                    Spearman_metric_bug = df_versoion_testing.loc[:, [m, 'bug']].astype(float).corr(method='spearman')
                    Pearson_metric_bug = 2 * np.sin(np.pi * Spearman_metric_bug[m][1] / 6)

                    s_p, s, f_p, f = 0, 0, 0, 0

                    # if tau_removed < 0:
                    if Pearson_metric_bug < 0:
                        for j in range(len(df_versoion_testing)):
                            # 过滤掉不在度量取值范围的阈值的类，即代入sloc值后超过度量取值范围的类过滤掉
                            if (auc_t - a_const - b_SLOC * df_versoion_testing.loc[j, 'SLOC'] > t_max) or \
                                    (auc_t - a_const - b_SLOC * df_versoion_testing.loc[j, 'SLOC'] < t_min):
                                df_versoion_testing = df_versoion_testing.drop(axis=0, index=j, inplace=False)
                                continue
                            if float(df_versoion_testing.loc[j, m]) <= \
                                    auc_t - a_const - b_SLOC * df_versoion_testing.loc[j, 'SLOC']:
                                df_versoion_testing.loc[j, 'predictBinary'] = 1
                                s += df_versoion_testing.loc[j, 'SLOC']
                                s_p += df_versoion_testing.loc[j, 'SLOC'] * 1
                                f += df_versoion_testing.loc[j, 'bug']
                                f_p += df_versoion_testing.loc[j, 'bug'] * 1
                            else:
                                df_versoion_testing.loc[j, 'predictBinary'] = 0
                                s += df_versoion_testing.loc[j, 'SLOC']
                                s_p += df_versoion_testing.loc[j, 'SLOC'] * 0
                                f += df_versoion_testing.loc[j, 'bug']
                                f_p += df_versoion_testing.loc[j, 'bug'] * 0
                    else:
                        for j in range(len(df_versoion_testing)):
                            # 过滤掉不在度量取值范围的阈值
                            if (auc_t - a_const - b_SLOC * df_versoion_testing.loc[j, 'SLOC'] > t_max) or \
                                    (auc_t - a_const - b_SLOC * df_versoion_testing.loc[j, 'SLOC'] < t_min):
                                # df_versoion_testing.drop(axis=0, index=j, inplace=True)
                                df_versoion_testing = df_versoion_testing.drop(axis=0, index=j, inplace=False)
                                continue
                            if float(df_versoion_testing.loc[j, m]) >= \
                                    auc_t - a_const - b_SLOC * df_versoion_testing.loc[j, 'SLOC']:
                                df_versoion_testing.loc[j, 'predictBinary'] = 1
                                s += df_versoion_testing.loc[j, 'SLOC']
                                s_p += df_versoion_testing.loc[j, 'SLOC'] * 1
                                f += df_versoion_testing.loc[j, 'bug']
                                f_p += df_versoion_testing.loc[j, 'bug'] * 1
                            else:
                                df_versoion_testing.loc[j, 'predictBinary'] = 0
                                s += df_versoion_testing.loc[j, 'SLOC']
                                s_p += df_versoion_testing.loc[j, 'SLOC'] * 0
                                f += df_versoion_testing.loc[j, 'bug']
                                f_p += df_versoion_testing.loc[j, 'bug'] * 0

                    # 所有类的去除混合效应后的阈值都不在度量取值范围，过滤掉
                    if s_p == 0 and s == 0 and f_p == 0 and f == 0:
                        continue

                    if f == 0:
                        effort_random == 0
                    else:
                        effort_random = f_p / f
                    effort_m = s_p / s
                    # ER_m = (effort_random - effort_m) / effort_random
                    if effort_random == 0:
                        ER_m = 0
                    else:
                        ER_m = (effort_random - effort_m) / effort_random

                    # confusion_matrix()函数中需要给出label, 0和1，否则该函数算不出TP,因为不知道哪个标签是poistive.
                    c_matrix = confusion_matrix(df_versoion_testing["bugBinary"], df_versoion_testing['predictBinary'],
                                                labels=[0, 1])
                    tn, fp, fn, tp = c_matrix.ravel()

                    if (tn + fp) == 0:
                        tnr_value = 0
                    else:
                        tnr_value = tn / (tn + fp)

                    if (fp + tn) == 0:
                        fpr = 0
                    else:
                        fpr = fp / (fp + tn)

                    bugBinary_value_counts = df_versoion_testing['bugBinary'].value_counts()
                    if len(bugBinary_value_counts) == 1:
                        continue
                    auc_value = roc_auc_score(df_versoion_testing['bugBinary'], df_versoion_testing['predictBinary'],
                                              labels=[0, 1])
                    # fpr_auc, tpr_auc, thresholds_auc = roc_curve(df_versoion_testing['bugBinary'],
                    #                                              df_versoion_testing['predictBinary'], pos_label=1)
                    # auc_value = auc(fpr_auc, tpr_auc)

                    recall_value = recall_score(df_versoion_testing['bugBinary'], df_versoion_testing['predictBinary'],
                                                labels=[0, 1])
                    precision_value = precision_score(df_versoion_testing['bugBinary'],
                                                      df_versoion_testing['predictBinary'], labels=[0, 1])
                    f1_value = f1_score(df_versoion_testing['bugBinary'], df_versoion_testing['predictBinary'],
                                        labels=[0, 1])

                    gm_value = (recall_value * tnr_value) ** 0.5
                    pfr = recall_value
                    pdr = fpr  # fp / (fp + tn)
                    bpp_value = 1 - (((0 - pfr) ** 2 + (1 - pdr) ** 2) * 0.5) ** 0.5

                    # print(auc_value, recall_value, precision_value, f1_value, gm_value, bpp_value)
                    dynamicPerformance = dynamicPerformance.append(
                        {'version': files[i], 'metric': m, 'ER': ER_m, 'auc_value': auc_value,
                         'recall_value': recall_value, 'precision_value': precision_value, 'f1_value': f1_value,
                         'gm_value': gm_value, 'bpp_value': bpp_value, 'auc_t': auc_t, 'a_const': a_const,
                         'b_SLOC': b_SLOC}, ignore_index=True)

                    dynamicPerformance.to_csv(result_directory + 'maxAUCthreshold\\dynamicPerformance_' + project
                                              + '.csv', index=False)
                    # break
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
    spv_result_Directory = "F:\\DT\\SPV\\"
    os.chdir(working_Directory)

    # 1. predict the defect-prone class on the consecutive release,i.e.,using the former threshold.
    static_threshold_spv(working_Directory, spv_result_Directory)

    # 2. for dynamic threshold
    # dynamic_threshold_spv(working_Directory, spv_result_Directory)

    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ".\nFrom ", time.asctime(time.localtime(s_time)), " to ",
          time.asctime(time.localtime(e_time)), ",\nThis", os.path.basename(sys.argv[0]), "ended within",
          execution_time, "(s), or ", (execution_time / 60), " (m).")
