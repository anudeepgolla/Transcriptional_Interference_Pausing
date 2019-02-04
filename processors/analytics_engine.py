import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


df = pd.read_csv('data/processed_genomic_data_set_v2.csv')
n, f_name, f_name_ver, mark_true, mark_false, mode, x_val, y_val = len(sys.argv) - 1, '', '', 3, 0.1, 0, '', ''
# print(sys.argv)
if n > 6 or n < 4:
    raise ValueError
elif n == 6:
    mode, f_name_ver, mark_true, mark_false, x_val, y_val = int(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), \
                                                            str(sys.argv[4]), str(sys.argv[5]), str(sys.argv[6])

    mark_true = 3 if mark_true == 'default' else float(mark_true)
    mark_false = 0.1 if mark_false == 'default' else float(mark_false)

    if mode > 0:
        f_name = '{}__&&__{}__strict_regression_{}'.format(x_val, y_val, f_name_ver)
    else:
        f_name = '{}__&&__{}__loose_regression_{}'.format(x_val, y_val, f_name_ver)

elif n == 5:
    mode, f_name_ver, mark_true, mark_false, x_val = int(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), \
                                                            str(sys.argv[4]), str(sys.argv[5])

    mark_true = 3 if mark_true == 'default' else float(mark_true)
    mark_false = 0.1 if mark_false == 'default' else float(mark_false)

    if mode > 0:
        f_name = '{}__&&__ALL_strict_regression_{}'.format(x_val, f_name_ver)
    else:
        f_name = '{}__&&__ALL__loose_regression_{}'.format(x_val, f_name_ver)

elif n == 4:
    mode, f_name_ver, mark_true, mark_false = int(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), \
                                                            str(sys.argv[4])

    mark_true = 3 if mark_true == 'default' else float(mark_true)
    mark_false = 0.1 if mark_false == 'default' else float(mark_false)

    if mode > 0:
        f_name = 'ALL_strict_regression_{}'.format(f_name_ver)
    else:
        f_name = 'ALL__loose_regression_{}'.format(f_name_ver)


print('Building {} Image ...'.format(f_name))
# print(df['position.1'].head(5))

# print(df.shape)
# print(df.nunique())
# print(df.columns)
# print(df.head(10))
# print(df.isna().sum())


# plt.plot(df['Unnamed: 0'], df['start_dist_rel'], 'bo', markersize=0.1)
# plt.show()



# 1) process dataset - (normalizing script)
# 2) do regressions to get initial analytical idea (analytic_processor)
# 3) build logistic regression model arch (log reg model)
# 4) build neural net arch (nn model)


# 5) do full logistic regression (HC True, False)
#     3.1) test this model on LC True and lower epsilon until all true
# 6) neural net (deep learning)
# 7) do progressive times series/rnn/something model
# 8) test this model on LC True and lower epsilon until all true


# all models train on [HC True, False], [ALL]
# test [HC True, False] model on LC True







# for i in range(len(df['gene'])):
#     if pd.isna(df['gene'][i]):
#         df.set_value(i, 'gene', 0)
#     else:
#         df.set_value(i, 'gene', 1)


# print(df['gene'].head(5))
# df_true = df.loc[df['pause_status'] == 'HC True', ['pause_G']]
# df_true_ind = df.loc[df['pause_status'] == 'HC True', ['position']]
#
# df_false = df.loc[df['pause_status'] == 'FALSE', ['pause_G']]
# df_false_ind = df.loc[df['pause_status'] == 'FALSE', ['position']]
#
# # print(df_false)
#
# plt.plot(df_true_ind, df_true, 'bo', markersize=4)
# plt.plot(df_false_ind, df_false, 'ro', markersize=0.2)
# plt.show()



num_params = ['position', 'pause_G', 'pause_C',  'context_G',
       'context_C', 'gene_start', 'gene_end', 'start_dist_abs',
       'start_dist_rel', 'end_dist_abs', 'end_dist_rel']

cat_params = ['ref_base', 'gene', 'trans_base', 'pause_seq', 'pause_context']




if n == 6:
    if mode > 0:
        df_true_y = df.loc[df['pause_status_hc_true'] == 1, [y_val]]
        df_true_x = df.loc[df['pause_status_hc_true'] == 1, [x_val]]
    else:
        df_true_y = df.loc[df['pause_status_false'] == 0, [y_val]]
        df_true_x = df.loc[df['pause_status_false'] == 0, [x_val]]

    df_false_y = df.loc[df['pause_status_false'] == 1, [y_val]]
    df_false_x = df.loc[df['pause_status_false'] == 1, [x_val]]

    plt.plot(df_true_x, df_true_y, 'bo', markersize=mark_true)
    plt.plot(df_false_x, df_false_y, 'ro', markersize=mark_false)
    plt.xlabel(x_val)
    plt.ylabel(y_val)

elif n == 5:
    splt_ct = 1
    for i in range(len(num_params)):
        if num_params[i] != x_val:

            if mode > 0:
                df_true_y = df.loc[df['pause_status_hc_true'] == 1, [num_params[i]]]
                df_true_x = df.loc[df['pause_status_hc_true'] == 1, [x_val]]
            else:
                df_true_y = df.loc[df['pause_status_false'] == 0, [num_params[i]]]
                df_true_x = df.loc[df['pause_status_false'] == 0, [x_val]]

            df_false_y = df.loc[df['pause_status_false'] == 1, [num_params[i]]]
            df_false_x = df.loc[df['pause_status_false'] == 1, [x_val]]

            plt.subplot(5, 2, splt_ct)
            plt.plot(df_true_x, df_true_y, 'bo', markersize=mark_true)
            plt.plot(df_false_x, df_false_y, 'ro', markersize=mark_false)
            plt.xlabel(x_val)
            plt.ylabel(num_params[i])

            print('{:02}% STEP {:02}/{}'.format(int((splt_ct*100/10)//1), splt_ct, 10))
            splt_ct += 1


elif n == 4:

    for i in range(len(num_params)):
        for j in range(len(num_params)):

            if mode > 0:
                df_true_y = df.loc[df['pause_status_hc_true'] == 1, [num_params[j]]]
                df_true_x = df.loc[df['pause_status_hc_true'] == 1, [num_params[i]]]
            else:
                df_true_y = df.loc[df['pause_status_false'] == 0, [num_params[j]]]
                df_true_x = df.loc[df['pause_status_false'] == 0, [num_params[i]]]

            df_false_y = df.loc[df['pause_status_false'] == 1, [num_params[j]]]
            df_false_x = df.loc[df['pause_status_false'] == 1, [num_params[i]]]

            plt.subplot(11, 11, 11 * i + j + 1)
            plt.plot(df_true_x, df_true_y, 'bo', markersize=3)
            plt.plot(df_false_x, df_false_y, 'ro', markersize=0.1)
            plt.xlabel(num_params[i])
            plt.ylabel(num_params[j])

            print('{:02}% STEP {:03}/{}'.format(int(((11 * i + j + 1) * 100 / 121) // 1), 11 * i + j + 1, 121))
            # plt.savefig('figures/{}__&&__{}.png'.format(num_params[i], num_params[j]))

else:
    raise RuntimeError


plt.savefig('figures/{}'.format(str(f_name)))
plt.show()
