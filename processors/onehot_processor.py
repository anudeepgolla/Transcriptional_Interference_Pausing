import numpy as np

"""

sequence_data_processor.py : transpose or flattens onehot seq data into 1D inputs

"""

def transpose():
    # load data
    X = np.load('../data/seq/special/X_data_onehot_no_spike_selective.npy')
    print(X.shape)
    print(X[0].shape)

    # transpose each submatrix individually and append
    X_new = []
    for i in X:
        X_new.append(np.transpose(i))

    # create new numpy array
    X_new = np.array(X_new)
    print(X_new.shape)

    # save
    np.save('../data/seq/special/X_data_onehot_transpose_no_spike_selective.npy', X_new)




def flatten():
    # load
    X = np.load('../data/seq/special/X_data_onehot_no_spike_selective.npy')
    print(X.shape)
    # print(X[0])

    # print(X[0].reshape((218*4,)))

    # reshape into flattened size
    print(X.reshape(7790, 218*4).shape)
    X = X.reshape(7790, 218*4)

    # save
    np.save('../data/seq/special/X_data_onehot_flat_no_spike_selective.npy', X)





