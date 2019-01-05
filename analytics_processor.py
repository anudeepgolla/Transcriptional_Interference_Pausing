import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('data/genomic_data_set_v1.csv')

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


df_true = df.loc[df['pause_status'] == 'HC True', ['pause_G']]
df_true_ind = df.loc[df['pause_status'] == 'HC True', ['position.1']]

df_false = df.loc[df['pause_status'] == 'FALSE', ['pause_G']]
df_false_ind = df.loc[df['pause_status'] == 'FALSE', ['position.1']]

# print(df_false)

plt.plot(df_true_ind, df_true, 'bo', markersize=4)
plt.plot(df_false_ind, df_false, 'ro', markersize=0.2)
plt.show()
