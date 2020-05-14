#!/usr/bin/env python

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

"""
prepare the data by segmenting it as follows:
- x_train : input features, (100 - test_partition) % of full dataset
- x_val   : input features, (test_partition/2) % of full dataset
- x_test  : input features, (test_partition/2) % of full dataset
- y_train : label,          (100 - test_partition) % of full dataset
- y_val   : label,          (test_partition/2) % of full dataset
- y_test  : label,          (test_partition/2) % of full dataset
"""
def prepare(filename, features, test_partition):
    # load the dataset
    dset = pd.read_csv(filename).values
    # our x variable is columns 0-features, all rows
    x = dset[:,0:features]
    # our y variable is the last column, all rows
    y = dset[:,features]
    # scale every input feature between 0 and 1 inclusive
    x_scaled = preprocessing.MinMaxScaler().fit_transform(x)
    # take (100 - test_partition)% of the data for training and test_partition% for validation and testing
    x_train, x_val_and_test, y_train, y_val_and_test = train_test_split(x_scaled, y, test_size=test_partition)
    # split valition and testing into separate partitions of equal size
    x_val, x_test, y_val, y_test = train_test_split(x_val_and_test, y_val_and_test, test_size=0.5)
    return x_train, x_val, x_test, y_train, y_val, y_test

"""
model architecture is as follows:
- input layer    : 10 features
- hidden layer 1 : 32 neurons, ReLU activation
- hidden layer 2 : 32 neurons, ReLU activation
- output layer   : 1 neuron,   sigmoid activation
"""
def model():
    # specify our model's architecture
    model = Sequential([
        Dense(32, activation='relu', input_shape=(10,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    # compile model with stochastic gradient descent optimizer
    # to determine the optimizing multipliers for the hidden layers
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    x_train, x_val, x_test, y_train, y_val, y_test = prepare('./housepricedata.csv', 10, 0.3)
    model = model()
    model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_val, y_val))
    print(model.evaluate(x_test, y_test)) # returns (loss, accuracy) tuple
