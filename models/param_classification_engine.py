import pandas as pd
import numpy as np
import sys
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



"""

param_classification_engine.py : all ensemble and probabilistic model code for training and testing

all models have same steps
1) create model instance by calling sklean module class
2) model.fit(X_train, y_train) - to train model
3) preds = model.pred(X_test) - run the model on test
4) get_metrics(preds, y_test) - to see how well it did
5) optional: do crossvalidation for more surity
6) optional: do hyperparameter test (commented out in linear svc)

"""



# _, df_file, target, log_file_name = tuple(sys.argv)
# df_file, target, log_file = str(df_file), str(target), '../logs/{}.log'.format(log_file_name)
# # sys.stdout = open(log_file, 'w')
#
# df = pd.read_csv(df_file)
# # df_file = '../data/seq/sequence_ds_kmers_6_given_spike_strict_v1.csv'
# # df = pd.read_csv(df_file)
# # target = 'pause_status'
# print(df.shape)
#
# y_train = df[target]
# df = df.drop([target], axis=1)
# # df = df[['position', 'pause_G', 'pause_C',
# #        'context_G', 'context_C', 'gene', 'start_dist_rel', 'end_dist_rel',
# #        'gene_len', 'ref_base_A', 'ref_base_C', 'ref_base_G', 'ref_base_T',
# #        'trans_base_A', 'trans_base_C', 'trans_base_G', 'trans_base_T']]
# rem = []
# for c in df.columns:
#     if c[:7] == 'Unnamed':
#            rem.append(c)
# df = df.drop(rem, axis=1)
# print(df.columns)
#
#
# if df_file.find('kmer') >= 0:
#        cv = CountVectorizer(ngram_range=(4,4))
#        kmers_data = list(df['pause_context_kmers'])
#        X_train = cv.fit_transform(kmers_data)
# else:
#        X_train = df




# get system arguments
_, data_dir, enc = tuple(sys.argv)
# load data file paths
X_data_path = '{}/X_data_{}_no_spike_selective.npy'.format(str(data_dir), str(enc))
y_data_path = '{}/y_data_{}_no_spike_selective.npy'.format(str(data_dir), str(enc))
# path file name to store print info
log_path = '../logs/encoding/{}'.format(str(enc))
# open the output path and start writing
sys.stdout = open(log_path, 'w')

# load data
X_train = np.load(X_data_path)
y_train = np.load(y_data_path)
print(X_train.shape)
print(y_train.shape)

# split data into training and testing (20%)
X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.2)


# return all neccesary analysis between two vectors
def get_metrics(y_test, y_predicted):
    # accuracy, precision, recall, f1, cohen-kappa
       accuracy = accuracy_score(y_test, y_predicted)
       precision = precision_score(y_test, y_predicted, average='weighted')
       recall = recall_score(y_test, y_predicted, average='weighted')
       f1 = f1_score(y_test, y_predicted, average='weighted')
       cohen_kappa = cohen_kappa_score(y_test, y_predicted)
       return accuracy, precision, recall, f1, cohen_kappa

def display_metrics(y_test, y_pred):
    # print confusion matrix ass well as stats from get_metrics()
       print("Confusion matrix\n")
       print(confusion_matrix(y_test, y_pred))
       print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
       accuracy, precision, recall, f1, cohen_kappa = get_metrics(y_test, y_pred)
       print("\naccuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f \ncohen_kappa = %.3f" % (accuracy, precision, recall, f1, cohen_kappa))



"""
LINEAR SVC MODEL
"""
# call, train, test a model
def linear_svc():
    print('############################## LIN SVC CLASSIFIER ##############################')
    # create an object with linear svc algorithm architecture
    svc = LinearSVC()
    # parameters_lin = {"fit_intercept" : [True, False], "normalize" : [True, False], "copy_X" : [True, False]}
    # grid_linreg = GridSearchCV(linreg, parameters_lin, verbose=1 , scoring = "r2")
    # grid_linreg.fit(X_training, y_training)
    #
    # print("Best LinReg Model: " + str(grid_linreg.best_estimator_))
    # print("Best Score: " + str(grid_linreg.best_score_))
    #
    # linreg = grid_linreg.best_estimator_

    # train model using training and labels
    svc.fit(X_training, y_training)

    # run the model on validation set 
    svc_pred = svc.predict(X_valid)

    # get accuracy score
    acc_svc = accuracy_score(y_valid, svc_pred)

    print(np.transpose((np.array(y_valid[:20]))))
    print(svc_pred[:20])

    print("Accuracy Score: " + str(acc_svc))

    # do cross validation test on data
    scores_svc = cross_val_score(svc, X_training, y_training, cv=10)
    print(scores_svc)
    print("Cross Validation Score: " + str(np.mean(scores_svc)))


"""
DECISION TREE MODEL
"""
def decision_tree():
    print('############################## DECISION TREE CLASSIFIER ##############################')
    dtc = DecisionTreeClassifier()
    dtc.fit(X_training, y_training)
    dtc_pred = dtc.predict(X_valid)
    print(np.transpose((np.array(y_valid[:20]))))
    print(dtc_pred[:20])
    display_metrics(y_valid, dtc_pred)
    scores_dtc = cross_val_score(dtc, X_training, y_training, cv=10)
    print(scores_dtc)
    print("Cross Validation Score: " + str(np.mean(scores_dtc)))





"""
FOREST MODEL
"""
def random_forest():
    print('############################## RANDOM FOREST CLASSIFIER ##############################')
    rfc = RandomForestClassifier()
    # parameters_lin = {"fit_intercept" : [True, False], "normalize" : [True, False], "copy_X" : [True, False]}
    # grid_linreg = GridSearchCV(linreg, parameters_lin, verbose=1 , scoring = "r2")
    # grid_linreg.fit(X_training, y_training)
    #
    # print("Best LinReg Model: " + str(grid_linreg.best_estimator_))
    # print("Best Score: " + str(grid_linreg.best_score_))
    #
    # linreg = grid_linreg.best_estimator_
    rfc.fit(X_training, y_training)
    rfc_pred = rfc.predict(X_valid)
    # acc_rfc = accuracy_score(y_valid, rfc_pred)

    print(np.transpose((np.array(y_valid[:20]))))
    print(rfc_pred[:20])

    display_metrics(y_valid, rfc_pred)

    # print("Accuracy Score: " + str(acc_rfc))

    scores_rfc = cross_val_score(rfc, X_training, y_training, cv=10)
    print(scores_rfc)
    print("Cross Validation Score: " + str(np.mean(scores_rfc)))


"""
ADABOOST MODEL
"""
def adaboost():
    print('############################## ADABOOST CLASSIFIER ##############################')
    abc = AdaBoostClassifier()
    abc.fit(X_training, y_training)
    abc_pred = abc.predict(X_valid)
    print(np.transpose((np.array(y_valid[:20]))))
    print(abc_pred[:20])
    display_metrics(y_valid, abc_pred)
    scores_abc = cross_val_score(abc, X_training, y_training, cv=10)
    print(scores_abc)
    print("Cross Validation Score: " + str(np.mean(scores_abc)))


"""
BAGGING TREES MODEL
"""
def bagging_trees():
    print('############################## BAGGING TREES CLASSIFIER ##############################')
    bgc = BaggingClassifier()
    bgc.fit(X_training, y_training)
    bgc_pred = bgc.predict(X_valid)
    print(np.transpose((np.array(y_valid[:20]))))
    print(bgc_pred[:20])
    display_metrics(y_valid, bgc_pred)
    scores_bgc = cross_val_score(bgc, X_training, y_training, cv=10)
    print(scores_bgc)
    print("Cross Validation Score: " + str(np.mean(scores_bgc)))



"""
EXTRA TREES MODEL
"""
def extra_trees():
    print('############################## EXTRA TREES CLASSIFIER ##############################')
    etc = ExtraTreesClassifier()
    etc.fit(X_training, y_training)
    etc_pred = etc.predict(X_valid)
    print(np.transpose((np.array(y_valid[:20]))))
    print(etc_pred[:20])
    display_metrics(y_valid, etc_pred)
    scores_etc = cross_val_score(etc, X_training, y_training, cv=10)
    print(scores_etc)
    print("Cross Validation Score: " + str(np.mean(scores_etc)))




"""
GRADIENT BOOSTING TREES MODEL
"""
def gradient_boosting():
    print('############################## GRADIENT BOOSTING TREES CLASSIFIER ##############################')
    gbc = GradientBoostingClassifier()
    gbc.fit(X_training, y_training)
    gbc_pred = gbc.predict(X_valid)
    print(np.transpose((np.array(y_valid[:20]))))
    print(gbc_pred[:20])
    display_metrics(y_valid, gbc_pred)
    scores_gbc = cross_val_score(gbc, X_training, y_training, cv=10)
    print(scores_gbc)
    print("Cross Validation Score: " + str(np.mean(scores_gbc)))




"""
NAIVE BAYES GAUSSIAN MODEL
"""
def naive_bayes_normal():
    print('############################## NAIVE BAYES CLASSIFIER ##############################')
    nbg = GaussianNB()
    nbg.fit(X_training, y_training)
    nbg_pred = rfc.predict(X_valid)
    print(np.transpose((np.array(y_valid[:20]))))
    print(nbg_pred[:20])
    display_metrics(y_valid, nbg_pred)
    scores_nbg = cross_val_score(nbg, X_training, y_training, cv=10)
    print(scores_nbg)
    print("Cross Validation Score: " + str(np.mean(scores_nbg)))



"""
NAIVE BAYES MULTINOMIAL MODEL
"""
def naive_bayes_multinomial():
    print('############################## NAIVE BAYES CLASSIFIER ##############################')
    mnb = MultinomialNB()
    mnb.fit(X_training, y_training)
    mnb_pred = mnb.predict(X_valid)
    print(np.transpose((np.array(y_valid[:20]))))
    print(mnb_pred[:20])
    display_metrics(y_valid, mnb_pred)
    scores_mnb = cross_val_score(mnb, X_training, y_training, cv=10)
    print(scores_mnb)
    print("Cross Validation Score: " + str(np.mean(scores_mnb)))






"""
K-NN MODEL
"""
def knn():
    print('############################## KNN CLASSIFIER ##############################')
    knn = KNeighborsClassifier()
    knn.fit(X_training, y_training)
    knn_pred = knn.predict(X_valid)
    print(np.transpose((np.array(y_valid[:20]))))
    print(knn_pred[:20])
    display_metrics(y_valid, knn_pred)
    scores_knn = cross_val_score(knn, X_training, y_training, cv=10)
    print(scores_knn)
    print("Cross Validation Score: " + str(np.mean(scores_knn)))


"""
Logisitc Regression MODEL
"""
def log_reg():
    print('############################## LOGISTIC REGRESSION ##############################')
    logreg = LogisticRegression()
    logreg.fit(X_training, y_training)
    logreg_pred = logreg.predict(X_valid)
    print(np.transpose((np.array(y_valid[:20]))))
    print(logreg_pred[:20])
    display_metrics(y_valid, logreg_pred)
    scores_logreg = cross_val_score(logreg, X_training, y_training, cv=10)
    print(scores_logreg)
    print("Cross Validation Score: " + str(np.mean(scores_logreg)))


"""
NEURAL NETWORK MODEL
"""
def nn():
    print('############################## NEURAL NETWORK CLASSIFIER ##############################')
    mlpc = MLPClassifier()
    mlpc.fit(X_training, y_training)
    mlpc_pred = mlpc.predict(X_valid)
    print(np.transpose((np.array(y_valid[:20]))))
    print(mlpc_pred[:20])
    display_metrics(y_valid, mlpc_pred)
    scores_mlpc = cross_val_score(mlpc, X_training, y_training, cv=10)
    print(scores_mlpc)
    print("Cross Validation Score: " + str(np.mean(scores_mlpc)))



"""
QDA MODEL
"""
def qda():
    print('############################## QDA CLASSIFIER ##############################')
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_training, y_training)
    qda_pred = qda.predict(X_valid)
    print(np.transpose((np.array(y_valid[:20]))))
    print(qda_pred[:20])
    display_metrics(y_valid, qda_pred)
    scores_qda = cross_val_score(qda, X_training, y_training, cv=10)
    print(scores_qda)
    print("Cross Validation Score: " + str(np.mean(scores_qda)))




# call models to run, comment out models not wanted to run

linear_svc()
decision_tree()
random_forest()
adaboost()
bagging_trees()
extra_trees()
gradient_boosting()
naive_bayes_normal()
naive_bayes_multinomial()
knn()
log_reg()
nn()
qda()




