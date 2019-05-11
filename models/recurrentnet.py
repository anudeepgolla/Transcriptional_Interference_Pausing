# LSTM for sequence classification in the IMDB dataset

import numpy
import sys
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, confusion_matrix
import os


"""

recurrentnet.py : code for LSTM model for training and testing

all models have same steps
1) create model instance by building it out using keras framework
2) model.fit(X_train, y_train) - to train model
3) preds = model.pred(X_test) - run the model on test
4) get_metrics(preds, y_test) - to see how well it did
5) optional: do crossvalidation for more surity
6) optional: do hyperparameter test (commented out in linear svc)

"""


os.environ['KMP_DUPLICATE_LIB_OK']='True'

# get this var from arugments in command line
run_type = str(sys.argv[1])

# set:
#   input_dim: depth of input sample
#   input_len: width of input sample
#   embedding_dim: what embedding space you want to encode in
#   X_load_file, y_load_file: file paths for data loading
#   eps: epochs - number of times to go through data during training
#   bsz: batch size: how many samples to look at every time you make adjustments to the model
#   log_file: file path to store output


if run_type == 'ordinal':
    input_dim = 4
    input_len = 218
    embedding_dim = 64
    X_load_file = '../data/seq/special/X_data_ordinal_no_spike_selective.npy'
    y_load_file = '../data/seq/special/y_data_ordinal_no_spike_selective.npy'
    eps = 10
    bsz = 64
    log_file = '../logs/recurrent/ordinal'

if run_type == 'onehot':
    input_dim = 2
    input_len = 218*4
    embedding_dim = 64
    X_load_file = '../data/seq/special/X_data_onehot_flat_no_spike_selective.npy'
    y_load_file = '../data/seq/special/y_data_onehot_flat_no_spike_selective.npy'
    eps = 10
    bsz = 64
    log_file = '../logs/recurrent/onehot_flat'

if run_type == 'kmers_4':
    input_dim = 219-4
    input_len = 4**4
    embedding_dim = 64
    X_load_file = '../data/seq/special/X_data_kmers_4_vectorized_no_spike_selective.npy'
    y_load_file = '../data/seq/special/y_data_kmers_4_vectorized_no_spike_selective.npy'
    eps = 10
    bsz = 64
    log_file = '../logs/recurrent/kmers_4.log'

if run_type == 'kmers_6':
    input_dim = 219-6
    input_len = 4**6
    embedding_dim = 64
    X_load_file = '../data/seq/special/X_data_kmers_6_vectorized_no_spike_selective.npy'
    y_load_file = '../data/seq/special/y_data_kmers_6_vectorized_no_spike_selective.npy'
    eps = 10
    bsz = 64
    log_file = '../logs/recurrent/kmers_6.log'

if run_type == 'kmers_8':
    input_dim = 219-8
    input_len = 4**8
    embedding_dim = 64
    X_load_file = '../data/seq/special/X_data_kmers_8_vectorized_no_spike_selective.npy'
    y_load_file = '../data/seq/special/y_data_kmers_8_vectorized_no_spike_selective.npy'
    eps = 10
    bsz = 64
    log_file = '../logs/recurrent/kmers_8.log'



# open output and starting writing
sys.stdout = open(log_file, 'w')
# load data
X_train = np.load(X_load_file)
y_train = np.load(y_load_file)
print(X_train.shape)
print(X_train[0])
print(y_train.shape)
print(y_train[0])

# split data into training and test
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

# get accuracy, precision, recall, f1, kappa
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    cohen_kappa = cohen_kappa_score(y_test, y_predicted)
    return accuracy, precision, recall, f1, cohen_kappa

# display metrics and confusion matrix
def display_metrics(y_test, y_pred):
    print("Confusion matrix\n")
    print(confusion_matrix(y_test, y_pred))
    print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
    accuracy, precision, recall, f1, cohen_kappa = get_metrics(y_test, y_pred)
    print("\naccuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f \ncohen_kappa = %.3f" % (
    accuracy, precision, recall, f1, cohen_kappa))


# create the model instance
model = Sequential()
# add embedding layer
model.add(Embedding(input_dim, embedding_dim, input_length=input_len))
# add LSTM (recurrent layer)
model.add(LSTM(100))
# add one regular neural layer with 1 node, activation sigmoid
model.add(Dense(1, activation='sigmoid'))
# use crossentropy loss, opimizer = adam (fancy gradient descent), loss on accuracy metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print stats of model architectur
print(model.summary())
# train model on training data for eps epochs and batch size bsz
model.fit(X_train, y_train, epochs=eps, batch_size=bsz)
# Final evaluation of the model, get stats results (this is built into keras, different from sklearn)
scores = model.evaluate(X_test, y_test, verbose=0)
print("MODEL Accuracy: %.2f%%" % (scores[1]*100))

# y_pred = model.predict(X_test)
# for i in range(len(y_pred)):
#     y_pred[i] = 0 if y_pred[i] < 0.5 else 1
# print(np.transpose((np.array(y_test[:20]))))
# print(y_pred[:20])
# display_metrics(y_test, y_pred)

