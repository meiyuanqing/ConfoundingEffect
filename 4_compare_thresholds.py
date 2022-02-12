#!/usr/bin/env python
# encoding:utf-8
"""
Author : Yuanqing Mei
Date : 2022/2/1
Time: 13:51
File: 4_compare_thresholds.py
HomePage : https://github.com/meiyuanqing
Email : dg1533019@smail.nju.edu.cn
"""


def compare_spv(comparing_dir):
    from scipy import stats
    import rpy2.robjects as robjects
    import pandas as pd
    import numpy as np

    # display all columns and rows, and set the item of row of dataframe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 5000)

    robjects.r('''
    library("rcompanion")
    library("effsize")
               ''')

    work_dir = comparing_dir
    dynamic_dir = comparing_dir
    static_dir = comparing_dir

    with open(work_dir + "List.txt") as l_all:
        lines_all = l_all.readlines()

    for line in lines_all:
        project = line.replace("\n", "")
        # if project != 'camel':
        #     continue
        print("the file is ", project)

        if not os.path.exists(dynamic_dir + 'maxAUCthreshold\\dynamicPerformance_' + project + '.csv'):
            continue

        df_dynamic = pd.read_csv(dynamic_dir + 'maxAUCthreshold\\dynamicPerformance_' + project + '.csv')

        df_static = pd.read_csv(static_dir + 'maxAUCthreshold\\staticPerformance_' + project + '.csv')
        # df_static_auc = pd.read_csv(static_dir + 'staticPerformance_auc_' + project + '.csv')
        # df_static_bpp = pd.read_csv(static_dir + 'staticPerformance_bpp_' + project + '.csv')
        # df_static_gm = pd.read_csv(static_dir + 'staticPerformance_gm_' + project + '.csv')
        # df_static_mfm = pd.read_csv(static_dir + 'staticPerformance_mfm_' + project + '.csv')

        cmp_dynamic_static = pd.DataFrame(
            columns=['metric', 'ER_dynamic_avg', 'ER_dynamic_var', 'ER_static_avg', 'ER_static_var', 'ER_wilcoxon',
                     'ER_cliff', 'auc_dynamic_avg', 'auc_dynamic_var', 'auc_static_avg', 'auc_static_var',
                     'auc_wilcoxon', 'auc_cliff', 'recall_dynamic_avg', 'recall_dynamic_var', 'recall_static_avg',
                     'recall_static_var', 'recall_wilcoxon', 'recall_cliff', 'precision_dynamic_avg',
                     'precision_dynamic_var', 'precision_static_avg', 'precision_static_var', 'precision_wilcoxon',
                     'precision_cliff', 'f1_dynamic_avg', 'f1_dynamic_var', 'f1_static_avg', 'f1_static_var',
                     'f1_wilcoxon', 'f1_cliff', 'gm_dynamic_avg', 'gm_dynamic_var', 'gm_static_avg', 'gm_static_var',
                     'gm_wilcoxon', 'gm_cliff', 'bpp_dynamic_avg', 'bpp_dynamic_var', 'bpp_static_avg',
                     'bpp_static_var', 'bpp_wilcoxon', 'bpp_cliff'])

        for m in df_dynamic['metric'].unique():
            print("the metric is ", m)
            dynamic_ER, static_ER = [], []
            dynamic_auc, static_auc = [], []
            dynamic_recall, static_recall = [], []
            dynamic_precision, static_precision = [], []
            dynamic_f1, static_f1 = [], []
            dynamic_gm, static_gm = [], []
            dynamic_bpp, static_bpp = [], []

            df_dynamic_m = df_dynamic[df_dynamic['metric'] == m].loc[:, :].reset_index()
            df_static_m = df_static[df_static['metric'] == m].loc[:, :].reset_index()

            for version in df_dynamic_m['version'].unique():

                # 过滤auc小于0.5
                if df_dynamic_m[df_dynamic_m['version'] == version].loc[:, 'auc_value'].tolist()[0] <= 0.5:
                    print("the version is ", version, " the auc value is ",
                          df_dynamic_m[df_dynamic_m['version'] == version].loc[:, 'auc_value'].tolist()[0] <= 0.5)
                    continue

                if version in df_static_m['version'].unique():
                    dynamic_ER.append(df_dynamic_m[df_dynamic_m['version'] == version].loc[:, 'ER'].tolist()[0])
                    dynamic_auc.append(df_dynamic_m[df_dynamic_m['version'] == version].loc[:, 'auc_value'].tolist()[0])
                    dynamic_recall.append(
                        df_dynamic_m[df_dynamic_m['version'] == version].loc[:, 'recall_value'].tolist()[0])
                    dynamic_precision.append(
                        df_dynamic_m[df_dynamic_m['version'] == version].loc[:, 'precision_value'].tolist()[0])
                    dynamic_f1.append(df_dynamic_m[df_dynamic_m['version'] == version].loc[:, 'f1_value'].tolist()[0])
                    dynamic_gm.append(df_dynamic_m[df_dynamic_m['version'] == version].loc[:, 'gm_value'].tolist()[0])
                    dynamic_bpp.append(df_dynamic_m[df_dynamic_m['version'] == version].loc[:, 'bpp_value'].tolist()[0])

                    static_ER.append(
                        df_static_m[df_static_m['version'] == version].loc[:, 'ER'].tolist()[0])
                    static_auc.append(
                        df_static_m[df_static_m['version'] == version].loc[:, 'auc_value'].tolist()[0])
                    static_recall.append(
                        df_static_m[df_static_m['version'] == version].loc[:, 'recall_value'].tolist()[0])
                    static_precision.append(
                        df_static_m[df_static_m['version'] == version].loc[:, 'precision_value'].tolist()[
                            0])
                    static_f1.append(
                        df_static_m[df_static_m['version'] == version].loc[:, 'f1_value'].tolist()[0])
                    static_gm.append(
                        df_static_m[df_static_m['version'] == version].loc[:, 'gm_value'].tolist()[0])
                    static_bpp.append(
                        df_static_m[df_static_m['version'] == version].loc[:, 'bpp_value'].tolist()[0])

            if len(dynamic_ER) < 3 or dynamic_ER == static_ER:
                ER_wilcoxon = '/'
                ER_cliff_estimate = '/'
            else:
                ER_wilcoxon = stats.wilcoxon(dynamic_ER, static_ER).pvalue
                dynamic_ER_name = ['dynamic'] * len(dynamic_ER)
                static_ER_name = ['static'] * len(static_ER)
                ER_cliff = robjects.r['cliff.delta'](d=robjects.FloatVector(dynamic_ER + static_ER),
                                                     f=robjects.StrVector(dynamic_ER_name + static_ER_name))
                ER_cliff_estimate = np.array(ER_cliff.rx('estimate')).flatten()[0]

            if len(dynamic_auc) < 3 or dynamic_auc == static_auc:
                auc_wilcoxon = '/'
                auc_cliff_estimate = '/'
            else:
                auc_wilcoxon = stats.wilcoxon(dynamic_auc, static_auc).pvalue
                dynamic_auc_name = ['dynamic'] * len(dynamic_auc)
                static_auc_name = ['static'] * len(static_auc)
                auc_cliff = robjects.r['cliff.delta'](d=robjects.FloatVector(dynamic_auc + static_auc),
                                                      f=robjects.StrVector(dynamic_auc_name + static_auc_name))
                auc_cliff_estimate = np.array(auc_cliff.rx('estimate')).flatten()[0]

            if len(dynamic_recall) < 3 or dynamic_recall == static_recall:
                recall_wilcoxon = '/'
                recall_cliff_estimate = '/'
            else:
                recall_wilcoxon = stats.wilcoxon(dynamic_recall, static_recall).pvalue
                dynamic_recall_name = ['dynamic'] * len(dynamic_recall)
                static_recall_name = ['static'] * len(static_recall)
                recall_cliff = robjects.r['cliff.delta'](d=robjects.FloatVector(dynamic_recall + static_recall),
                                                         f=robjects.StrVector(dynamic_recall_name + static_recall_name))
                recall_cliff_estimate = np.array(recall_cliff.rx('estimate')).flatten()[0]

            if len(dynamic_precision) < 3 or dynamic_precision == static_precision:
                precision_wilcoxon = '/'
                precision_cliff_estimate = '/'
            else:
                precision_wilcoxon = stats.wilcoxon(dynamic_precision, static_precision).pvalue
                dynamic_precision_name = ['dynamic'] * len(dynamic_precision)
                static_precision_name = ['static'] * len(static_precision)
                precision_cliff = robjects.r['cliff.delta'](d=robjects.FloatVector(dynamic_precision + static_precision),
                                                         f=robjects.StrVector(dynamic_precision_name + static_precision_name))
                precision_cliff_estimate = np.array(precision_cliff.rx('estimate')).flatten()[0]

            cmp_dynamic_static = cmp_dynamic_static.append(
                {'metric': m, 'ER_dynamic_avg': np.mean(dynamic_ER), 'ER_dynamic_var': np.var(dynamic_ER),
                 'ER_static_avg': np.mean(static_ER), 'ER_static_var': np.mean(static_ER),
                 'ER_wilcoxon': ER_wilcoxon, 'ER_cliff': ER_cliff_estimate,
                 'auc_dynamic_avg': np.mean(dynamic_auc), 'auc_dynamic_var': np.var(dynamic_auc),
                 'auc_static_avg': np.mean(static_auc), 'auc_static_var': np.mean(static_auc),
                 'auc_wilcoxon': auc_wilcoxon, 'auc_cliff': auc_cliff_estimate,
                 'recall_dynamic_avg': np.mean(dynamic_recall), 'recall_dynamic_var': np.var(dynamic_recall),
                 'recall_static_avg': np.mean(static_recall), 'recall_static_var': np.mean(static_recall),
                 'recall_wilcoxon': recall_wilcoxon, 'recall_cliff': recall_cliff_estimate,
                 'precision_dynamic_avg': np.mean(dynamic_precision), 'precision_dynamic_var': np.var(dynamic_precision),
                 'precision_static_avg': np.mean(static_precision), 'precision_static_var': np.mean(static_precision),
                 'precision_wilcoxon': precision_wilcoxon, 'precision_cliff': precision_cliff_estimate

                 }, ignore_index=True)

            cmp_dynamic_static.to_csv(work_dir + 'maxAUCthreshold\\cmp_dynamic_static_' + project + '.csv', index=False)

            # break

        # break


if __name__ == '__main__':
    import os
    import sys
    import time
    import pandas as pd
    import numpy as np

    s_time = time.time()

    comparing_result = "F:\\DT\\SPV\\"

    os.chdir(comparing_result)

    compare_spv(comparing_result)

    e_time = time.time()
    execution_time = e_time - s_time

    print("The __name__ is ", __name__, ".\nFrom ", time.asctime(time.localtime(s_time)), " to ",
          time.asctime(time.localtime(e_time)), ",\nThis", os.path.basename(sys.argv[0]), "ended within",
          execution_time, "(s), or ", (execution_time / 60), " (m).")