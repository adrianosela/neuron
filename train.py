#!/usr/bin/env python

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

def prepared():
    """
    prepare the data by segmenting it as follows:
    - x_train : 10 input features, 70% of full dataset
    - x_val   : 10 input features, 15% of full dataset
    - x_test  : 10 input features, 15% of full dataset
    - y_train : 1 label,           70% of full dataset
    - y_val   : 1 label,           15% of full dataset
    - y_test  : 1 label,           15% of full dataset
    """

    # load the dataset
    df = pd.read_csv("./housepricedata.csv")
    dset = df.values

    # our x variable is columns 0-9 (all rows)
    # our y variable is column 10   (all rows)
    x, y = dset[:,0:10], dset[:,10]

    # scale the input features (x) such that
    # all values lie between 0 and 1 inclusive
    mms = preprocessing.MinMaxScaler()
    x_scaled = mms.fit_transform(x)

    # split training, validation, and test data
    x_train, x_val_and_test, y_train, y_val_and_test = train_test_split(x_scaled, y, test_size=0.3)
    x_val, x_test, y_val, y_test = train_test_split(x_val_and_test, y_val_and_test, test_size=0.5)

    return x_train, x_val, x_test, y_train, y_val, y_test

def model():
    """
    our model will be as follows:
    - input layer    : 10 features
    - hidden layer 1 : 32 neurons, ReLU activation
    - hidden layer 2 : 32 neurons, ReLU activation
    - output layer   : 1 neuron,   sigmoid activation
    """
    # specify our model's architecture
    model = Sequential([
        Dense(32, activation='relu', input_shape=(10,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])

    # compile model with stochastic gradient descent optimizer
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    x_train, x_val, x_test, y_train, y_val, y_test = prepared()
    model = model()

    hist = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_val, y_val))
