#!/usr/bin/env python

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

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

# segment our data as follows
"""
x_train : 10 input features, 70% of full dataset
x_val   : 10 input features, 15% of full dataset
x_test  : 10 input features, 15% of full dataset
y_train : 1 label,           70% of full dataset
y_val   : 1 label,           15% of full dataset
y_test  : 1 label,           15% of full dataset
"""
x_train, x_val_and_test, y_train, y_val_and_test = train_test_split(x_scaled, y, test_size=0.3)
x_val, x_test, y_val, y_test = train_test_split(x_val_and_test, y_val_and_test, test_size=0.5)

print(x_train.shape, x_val.shape, x_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)
