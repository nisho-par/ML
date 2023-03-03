# TO FIX

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import copy
import seaborn as sns

from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from dataset import df


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def get_xy(dataframe, y_label, x_labels=None):
    dataframe = copy.deepcopy(dataframe)
    if x_labels is None:
        X = dataframe[[c for c in dataframe.columns if c!=y_label]].values
    else:
        if len(x_labels) == 1:
            X = dataframe[x_labels[0]].values.reshape(-1, 1)
        else:
            X = dataframe[x_labels].values
    y = dataframe[y_label].values.reshape(-1, 1)
    data = np.hstack((X, y))
    
    return data, X, y

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
_, X_train_temp, y_train_temp = get_xy(train, 'Rented Bike Count', x_labels=['Temperature(°C)'])
_, X_valid_temp, y_valid_temp = get_xy(valid, 'Rented Bike Count', x_labels=['Temperature(°C)'])
_, X_test_temp, y_test_temp = get_xy(test, 'Rented Bike Count', x_labels=['Temperature(°C)'])

# neural network
temp_normalizer = tf.keras.layers.Normalization( input_shape=(1,), axis=None)
temp_normalizer.adapt(X_train_temp.reshape(-1))

nn_model = tf.keras.Sequential([
    temp_normalizer, 
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='relu')
])
nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

history = nn_model.fit( 
    X_train_temp, y_train_temp,
    verbose=0,
    epochs=100,
    validation_data=(X_valid_temp, y_valid_temp)
)

plot_loss(history)

plt.scatter(X_train_temp, y_train_temp, label='Data', color='blue')
x = tf.linspace(-20, 40, 100)
plt.plot(x, nn_model.predict(np.array(x).reshape(-1, 1)), label='Fit', color='red', linewidth=3)
plt.legend()
plt.title('bikes vs neural network')
plt.ylabel('number of bikes')
plt.xlabel('temp')
plt.show()
