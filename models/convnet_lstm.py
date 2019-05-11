from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, MaxPooling1D
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, confusion_matrix
import os


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
    log_file = '../logs/convnet_1d_lstm/ordinal'

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
    log_file = '../logs/convnet_1d_lstm/onehot_flat'

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
    log_file = '../logs/convnet_1d_lstm/kmers_4.log'

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
    log_file = '../logs/convnet_1d_lstm/kmers_6.log'

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
    log_file = '../logs/convnet_1d_lstm/kmers_8.log'

# Convolution
pool_size = 4
lstm_output_size = 64



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
model.add(Embedding(input_dim, embedding_dim, input_length=input_len))
# add dropout ( choose random nodes to stop learning )
model.add(Dropout(0.25))
# add convolution layer
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# add pooling layer
model.add(MaxPooling1D(pool_size=pool_size))
# add recurrent LSTM layer
model.add(LSTM(lstm_output_size))
# add regular neural layer with 1 node, sigmoid activation
model.add(Dense(1))
model.add(Activation('sigmoid'))

# loss is crossentropy, adam opt, learn on accuracy
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
# train
model.fit(X_train, y_train,
          batch_size=bsz,
          epochs=epochs,
          validation_data=(X_test, y_test))
# get stats
score, acc = model.evaluate(X_test, y_test, batch_size=bsz)
print('Test score:', score)
print('Test accuracy:', acc)