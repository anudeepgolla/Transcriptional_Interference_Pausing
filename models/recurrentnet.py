# LSTM for sequence classification in the IMDB dataset

import numpy
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

os.environ['KMP_DUPLICATE_LIB_OK']='True'



input_dim = 4
input_len = 218
embedding_dim = 64
X_load_file = '../data/seq/special/X_data_ordinal_no_spike_selective.npy'
y_load_file = '../data/seq/special/y_data_ordinal_no_spike_selective.npy'
eps = 3
bsz = 64

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


# create the model
model = Sequential()
model.add(Embedding(input_dim, embedding_dim, input_length=input_len))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=eps, batch_size=bsz)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("MODEL Accuracy: %.2f%%" % (scores[1]*100))

# y_pred = model.predict(X_test)
# for i in range(len(y_pred)):
#     y_pred[i] = 0 if y_pred[i] < 0.5 else 1
# print(np.transpose((np.array(y_test[:20]))))
# print(y_pred[:20])
# display_metrics(y_test, y_pred)

