from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, confusion_matrix
import os


"""

convent_1d.py : code for LSTM model for training and testing

all models have same steps
1) create model instance by building it out using keras framework
2) model.fit(X_train, y_train) - to train model
3) preds = model.pred(X_test) - run the model on test
4) get_metrics(preds, y_test) - to see how well it did
5) optional: do crossvalidation for more surity
6) optional: do hyperparameter test (commented out in linear svc)

"""


run_type = str(sys.argv[1])

# set:
#   input_dim: depth of input sample
#   input_len: width of input sample
#   embedding_dim: what embedding space you want to encode in
#   X_load_file, y_load_file: file paths for data loading
#   eps: epochs - number of times to go through data during training
#   bsz: batch size: how many samples to look at every time you make adjustments to the model
#   log_file: file path to store output
#   filters = complexity of convolution alyer
#   kernel_size = have close together to search (usually fixed),  found by variying until optimal found
#   hidden_dims = number of nodes in a midle neural layer



if run_type == 'ordinal':
    input_dim = 4
    input_len = 218
    X_load_file = '../data/seq/special/X_data_ordinal_no_spike_selective.npy'
    y_load_file = '../data/seq/special/y_data_ordinal_no_spike_selective.npy'
    bsz = 16
    embedding_dim = 64
    filters = 64
    kernel_size = 4
    hidden_dims = 64
    epochs = 10
    log_file = '../logs/convnet_1d/ordinal'

if run_type == 'onehot':
    input_dim = 2
    input_len = 218*4
    X_load_file = '../data/seq/special/X_data_onehot_flat_no_spike_selective.npy'
    y_load_file = '../data/seq/special/y_data_onehot_flat_no_spike_selective.npy'
    bsz = 16
    embedding_dim = 64
    filters = 256
    kernel_size = 4
    hidden_dims = 256
    epochs = 10
    log_file = '../logs/convnet_1d/onehot_flat'

if run_type == 'kmers_4':
    input_dim = 219-4
    input_len = 4**4
    X_load_file = '../data/seq/special/X_data_kmers_4_vectorized_no_spike_selective.npy'
    y_load_file = '../data/seq/special/y_data_kmers_4_vectorized_no_spike_selective.npy'
    bsz = 16
    embedding_dim = 64
    filters = 256
    kernel_size = 4
    hidden_dims = 256
    epochs = 10
    log_file = '../logs/convnet_1d/kmers_4.log'

if run_type == 'kmers_6':
    input_dim = 219-6
    input_len = 4**6
    X_load_file = '../data/seq/special/X_data_kmers_6_vectorized_no_spike_selective.npy'
    y_load_file = '../data/seq/special/y_data_kmers_6_vectorized_no_spike_selective.npy'
    bsz = 16
    embedding_dim = 64
    filters = 256
    kernel_size = 4
    hidden_dims = 256
    epochs = 10
    log_file = '../logs/convnet_1d/kmers_6.log'

if run_type == 'kmers_8':
    input_dim = 219-8
    input_len = 4**8
    X_load_file = '../data/seq/special/X_data_kmers_8_vectorized_no_spike_selective.npy'
    y_load_file = '../data/seq/special/y_data_kmers_8_vectorized_no_spike_selective.npy'
    bsz = 16
    embedding_dim = 64
    filters = 256
    kernel_size = 4
    hidden_dims = 256
    epochs = 10
    log_file = '../logs/convnet_1d/kmers_8.log'




sys.stdout = open(log_file, 'w')
X_train = np.load(X_load_file)
y_train = np.load(y_load_file)
print(X_train.shape)
print(X_train[0])
print(y_train.shape)
print(y_train[0])

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)


def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    cohen_kappa = cohen_kappa_score(y_test, y_predicted)
    return accuracy, precision, recall, f1, cohen_kappa


def display_metrics(y_test, y_pred):
    print("Confusion matrix\n")
    print(confusion_matrix(y_test, y_pred))
    print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
    accuracy, precision, recall, f1, cohen_kappa = get_metrics(y_test, y_pred)
    print("\naccuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f \ncohen_kappa = %.3f" % (
    accuracy, precision, recall, f1, cohen_kappa))


# model instance
model = Sequential()
# add embedding layer
model.add(Embedding(input_dim,
                    embedding_dim,
                    input_length=input_len))
# model.add(Dropout(0.2))

# add one Convolutional layer with 
#   below hyperparameters 
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))

# add pooling layer
model.add(GlobalMaxPooling1D())

# add regular neural layer with hidden_dims nodes with relu activation
model.add(Dense(hidden_dims))
# model.add(Dropout(0.2))
model.add(Activation('relu'))

# add nerual layer with 1 node for prediction with sigmoid activation
model.add(Dense(1))
model.add(Activation('sigmoid'))

# use crossentropy loss, adam optimizer, learn on accuracy
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# train
model.fit(X_train, y_train,
          batch_size=bsz,
          epochs=epochs,
          validation_data=(X_test, y_test))


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("MODEL Accuracy: %.2f%%" % (scores[1]*100))