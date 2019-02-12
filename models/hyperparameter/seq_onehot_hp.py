import pandas as pd
import numpy as np
import sys
import pickle
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, f_regression




# _, data_path, model_run_name = tuple(sys.argv)

# data_path = '../../data/param/param_ds_prcs_selectivelabel_sample_adj_v1.csv'
X_data_path = '../../data/seq/special/X_data_onehot_flat_no_spike_selective.npy'
y_data_path = '../../data/seq/special/y_data_onehot_flat_no_spike_selective.npy'
hp_cv = 2

# model_run_name = 'seq_onehot_final_v1_max_depth'
# parameters_etc = {'n_estimators': [10, 30, 50, 70, 90, 110, 130, 150],
#                   # 'criterion': ['gini', 'entropy'],
#                   'max_depth': [10, 30, 50, 70, 90, 110, None],
#                   # 'min_samples_split': [0.0001, 0.001, 0.01, 0.1],
#                   # 'min_samples_leaf': [0.0001, 0.001, 0.01, 0.1],
#                   # 'max_features': [None, 'sqrt', 'log2'],
#                   'verbose': [2]}

# model_run_name = 'seq_onehot_final_v1_min_samples_split'
# parameters_etc = {'n_estimators': [10, 30, 50, 70, 90, 110, 130, 150],
#                   # 'criterion': ['gini', 'entropy'],
#                   # 'max_depth': [10, 30, 50, 70, 90, 110, None],
#                   'min_samples_split': [0.0001, 0.001, 0.01, 0.1],
#                   # 'min_samples_leaf': [0.0001, 0.001, 0.01, 0.1],
#                   # 'max_features': [None, 'sqrt', 'log2'],
#                   'verbose': [2]}
#
# model_run_name = 'seq_onehot_final_v1_max_samples_leaf'
# parameters_etc = {'n_estimators': [10, 30, 50, 70, 90, 110, 130, 150],
#                   # 'criterion': ['gini', 'entropy'],
#                   # 'max_depth': [10, 30, 50, 70, 90, 110, None],
#                   # 'min_samples_split': [0.0001, 0.001, 0.01, 0.1],
#                   'min_samples_leaf': [0.0001, 0.001, 0.01, 0.1],
#                   # 'max_features': [None, 'sqrt', 'log2'],
#                   'verbose': [2]}


model_run_name = 'seq_onehot_final_tuned_reg3'
parameters_etc = {'n_estimators': [1],
                  # 'criterion': ['gini', 'entropy'],
                  'max_depth': [2],
                  'min_samples_split': [0.001],
                  'min_samples_leaf': [0.01],
                  'max_features': ['log2'],
                  'verbose': [2]}



log_path = '../../logs/hyperparameter/tuned/{}.log'.format(str(model_run_name))
model_path = '../../models/model_files/tuned/{}.sav'.format(str(model_run_name))
sys.stdout = open(log_path, 'w')

X_train = np.load(X_data_path)
y_train = np.load(y_data_path)
print(X_train.shape)
print(y_train.shape)
########################
# df = pd.read_csv(data_path)
# print(df.columns)
# rem_cols = []
# for c in df.columns:
#     if c[:7] == 'Unnamed':
#         rem_cols.append(c)
# df = df.drop(rem_cols, axis=1)
# print(df.columns)
#
#
# df_y = df['pause_status']
# y_train = np.array(df['pause_status'])
# df = df.drop(['pause_status'], axis=1)
# X_train = np.array(df)
#########################

# selector = SelectKBest(f_regression, k='all').fit(df, df_y)
selector = SelectKBest(f_regression, k='all').fit(X_train, y_train)
scores = selector.scores_
# print(scores)
# print(type(scores[5]), scores[5], scores[5] == 'nan')
# val = str(scores[5])
# print(type(val), val, val == 'nan')
########################
# score_dict = {}
# ct = 0
# for si in range(len(scores)):
#     if str(scores[si]) == 'nan':
#         scores[si] = 0
# for s in scores:
#     score_dict[s] = ct
#     ct += 1

# fin_order = []
# scores = sorted(scores, reverse=True)
# for s in scores:
#     if s:
#         print(score_dict[s])
#         fin_order.append(str(df.columns[score_dict[s]]))
#
# print("\nFEATURES SELECTKBEST PARAMS\n")
# print(fin_order[:])

######################
print("\nFEATURES SELECTKBEST SCORES\n")
print(scores)



X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.2)


def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    cohen_kappa = cohen_kappa_score(y_test, y_predicted)
    return accuracy, precision, recall, f1, cohen_kappa


def display_metrics(y_test, y_pred):
    print("\nConfusion matrix\n")
    print(confusion_matrix(y_test, y_pred))
    print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
    accuracy, precision, recall, f1, cohen_kappa = get_metrics(y_test, y_pred)
    print("\naccuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f \ncohen_kappa = %.3f" % (
    accuracy, precision, recall, f1, cohen_kappa))






print('\n############################## EXTRA TREES CLASSIFIER ##############################\n')
etc = ExtraTreesClassifier()
grid_etc = GridSearchCV(etc, parameters_etc, verbose=1, scoring="f1", cv=hp_cv)
grid_etc.fit(X_training, y_training)

grid_etc_scores = grid_etc.cv_results_
grid_etc_params = grid_etc.best_params_
print('\nCV RESULTS\n')
print(grid_etc_scores)
print('\nBEST PARAMETERS\n')
print(grid_etc_params)

print("\nBest Extratrees Model: " + str(grid_etc.best_estimator_))
print("\nBest Score: " + str(grid_etc.best_score_))

linreg = grid_etc.best_estimator_
etc.fit(X_training, y_training)

pickle.dump(etc, open(model_path, 'wb'))

etc_pred = etc.predict(X_valid)
etc_estimators = etc.estimators_
etc_feature_scores = etc.feature_importances_
print('\nESTIMATORS\n')
print(etc_estimators)
print('\nFEATURE SCORES\n')
print(etc_feature_scores)

print("\nMETRICS")
display_metrics(y_valid, etc_pred)

