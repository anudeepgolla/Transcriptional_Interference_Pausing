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


model = Sequential()
model.add(Embedding(input_dim, embedding_dim, input_length=input_len))
model.add(Dropout(0.25))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train,
          batch_size=bsz,
          epochs=epochs,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test, batch_size=bsz)
print('Test score:', score)
print('Test accuracy:', acc)