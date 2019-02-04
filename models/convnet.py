import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
#############################################

X_load_file = '../data/seq/special/X_data_onehot_no_spike_selective.npy'
y_load_file = '../data/seq/special/y_data_onehot_no_spike_selective.npy'
eps = 3
bsz = 64

X_train = np.load(X_load_file)
y_train = np.load(y_load_file)
print(X_train.shape)
print(X_train[0])
print(y_train.shape)
print(y_train[0])

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)


# # 2. Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print(X_train.shape)
print(y_train.shape)

# # 3. Preprocess class labels; i.e. convert 1-dimensional class arrays to 3-dimensional class matrices
# Y_train = np_utils.to_categorical(y_train, 2)
# Y_test = np_utils.to_categorical(y_test, 2)

# 4. Define model architecture
model = Sequential()
model.add(Conv2D(64, 1, 7, activation='relu'))
model.add(Conv2D(64, 1, 7, activation='relu'))
model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# 5. Compile model
model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

# 6. Fit model on training data
model.fit(X_train, y_train, validation_data=(X_test, y_test),
              batch_size=bsz, nb_epoch=eps, verbose=2)

# 7. Evaluate model on test data
score = model.evaluate(X_test, y_test, verbose=2)
print("score = " + str(score))