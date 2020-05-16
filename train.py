#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sys

def prepare(filename, features, test_partition):
    """
    prepare the data by segmenting it as follows:
    - x_train : input features, (100 - test_partition) % of full dataset
    - x_val   : input features, (test_partition/2) % of full dataset
    - x_test  : input features, (test_partition/2) % of full dataset
    - y_train : label,          (100 - test_partition) % of full dataset
    - y_val   : label,          (test_partition/2) % of full dataset
    - y_test  : label,          (test_partition/2) % of full dataset
    """

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

def build_model(features):
    """
    model architecture is as follows:
    - input layer    : features
    - hidden layer 1 : 32 neurons, ReLU activation
    - hidden layer 2 : 32 neurons, ReLU activation
    - output layer   : 1 neuron,   sigmoid activation
    """

    # specify our model's architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation=tf.nn.relu, input_shape=(features,)),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ])
    # compile model with stochastic gradient descent optimizer
    # to determine the optimizing multipliers for the hidden layers
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate(model, x_test, y_test):
    # evalurate returns a (loss, accuracy) tuple
    print(model.evaluate(x_test, y_test))

def plot_hist(hist):
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('loss')
    plt.ylabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()

if __name__ == "__main__":
    # parse and validate args
    try:
        filename, features, test_partition = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    except:
        print("\nerror - bad arguments\n")
        print("USAGE: ./train.py [FILENAME] [FEATURES] [TEST_PARTITION]")
        print("\ne.g. for a csv file with 10 features, taking 30% for testing the model")
        print("$ ./train.py ./house_prices.csv 10 0.3")
        sys.exit(1)

    # build the model
    model = build_model(features)
    # load the data and partition it into { training, validation, testing }
    x_train, x_val, x_test, y_train, y_val, y_test = prepare(filename, features, test_partition)
    # train the model
    hist = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_val, y_val))
    # visualize loss
    plot_hist(hist)

