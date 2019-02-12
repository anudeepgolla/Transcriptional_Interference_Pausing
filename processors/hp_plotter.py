import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# model_cv_results = []
# grid_param_1, grid_param_2, name_param_1, name_param_2 = '', '', '', ''
#
# def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
#     # Get Test Scores Mean and std for each grid search
#     scores_mean = cv_results['mean_test_score']
#     scores_mean = np.np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))
#
#     scores_sd = cv_results['std_test_score']
#     scores_sd = np.np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))
#
#     # Plot Grid search scores
#     _, ax = plt.subplots(1,1)
#
#     # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
#     for idx, val in enumerate(grid_param_2):
#         ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))
#
#     ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
#     ax.set_xlabel(name_param_1, fontsize=16)
#     ax.set_ylabel('CV Average Score', fontsize=16)
#     ax.legend(loc="best", fontsize=15)
#     ax.grid('on')
#
# # Calling Method
# plot_grid_search(pipe_grid.cv_results_, n_estimators, max_features, 'N Estimators', 'Max Features')

results = [{'min_samples_leaf': 0.0001, 'n_estimators': 10, 'verbose': 2}, {'min_samples_leaf': 0.0001, 'n_estimators': 30, 'verbose': 2}, {'min_samples_leaf': 0.0001, 'n_estimators': 50, 'verbose': 2}, {'min_samples_leaf': 0.0001, 'n_estimators': 70, 'verbose': 2}, {'min_samples_leaf': 0.0001, 'n_estimators': 90, 'verbose': 2}, {'min_samples_leaf': 0.0001, 'n_estimators': 110, 'verbose': 2}, {'min_samples_leaf': 0.0001, 'n_estimators': 130, 'verbose': 2}, {'min_samples_leaf': 0.0001, 'n_estimators': 150, 'verbose': 2}, {'min_samples_leaf': 0.001, 'n_estimators': 10, 'verbose': 2}, {'min_samples_leaf': 0.001, 'n_estimators': 30, 'verbose': 2}, {'min_samples_leaf': 0.001, 'n_estimators': 50, 'verbose': 2}, {'min_samples_leaf': 0.001, 'n_estimators': 70, 'verbose': 2}, {'min_samples_leaf': 0.001, 'n_estimators': 90, 'verbose': 2}, {'min_samples_leaf': 0.001, 'n_estimators': 110, 'verbose': 2}, {'min_samples_leaf': 0.001, 'n_estimators': 130, 'verbose': 2}, {'min_samples_leaf': 0.001, 'n_estimators': 150, 'verbose': 2}, {'min_samples_leaf': 0.01, 'n_estimators': 10, 'verbose': 2}, {'min_samples_leaf': 0.01, 'n_estimators': 30, 'verbose': 2}, {'min_samples_leaf': 0.01, 'n_estimators': 50, 'verbose': 2}, {'min_samples_leaf': 0.01, 'n_estimators': 70, 'verbose': 2}, {'min_samples_leaf': 0.01, 'n_estimators': 90, 'verbose': 2}, {'min_samples_leaf': 0.01, 'n_estimators': 110, 'verbose': 2}, {'min_samples_leaf': 0.01, 'n_estimators': 130, 'verbose': 2}, {'min_samples_leaf': 0.01, 'n_estimators': 150, 'verbose': 2}, {'min_samples_leaf': 0.1, 'n_estimators': 10, 'verbose': 2}, {'min_samples_leaf': 0.1, 'n_estimators': 30, 'verbose': 2}, {'min_samples_leaf': 0.1, 'n_estimators': 50, 'verbose': 2}, {'min_samples_leaf': 0.1, 'n_estimators': 70, 'verbose': 2}, {'min_samples_leaf': 0.1, 'n_estimators': 90, 'verbose': 2}, {'min_samples_leaf': 0.1, 'n_estimators': 110, 'verbose': 2}, {'min_samples_leaf': 0.1, 'n_estimators': 130, 'verbose': 2}, {'min_samples_leaf': 0.1, 'n_estimators': 150, 'verbose': 2}]


scores = [0.88105395, 0.88542624, 0.88653783, 0.88783639, 0.8908548 ,
       0.88912695, 0.88889494, 0.88579436, 0.71961146, 0.73111075,
       0.73717334, 0.74573543, 0.74763407, 0.74318132, 0.74438934,
       0.74457428, 0.51867071, 0.53686419, 0.52187367, 0.53647745,
       0.53628005, 0.53425749, 0.53220167, 0.5300809 , 0.40816931,
       0.43151711, 0.4245056 , 0.41649901, 0.42548193, 0.42564017,
       0.43537968, 0.43560078]

ct = 0
for i in range(32):
       # if results[i]['criterion'] == 'entropy' and results[i]['max_features'] == 'sqrt':
              print(i, results[i], scores[i])
              ct +=1

print(ct)
