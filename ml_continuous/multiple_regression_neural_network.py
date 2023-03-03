# to fix

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
_, X_train_all, y_train_all = get_xy(train, 'Rented Bike Count', x_labels=df.columns[1:])
_, X_valid_all, y_valid_all = get_xy(valid, 'Rented Bike Count', x_labels=df.columns[1:])
_, X_test_all, y_test_all = get_xy(test, 'Rented Bike Count', x_labels=df.columns[1:])

all_reg = LinearRegression()
all_reg.fit(X_train_all, y_train_all)

print(all_reg.coef_, all_reg.intercept_, all_reg.score(X_test_all, y_test_all))


# multiple regression with neural network
all_normalizer = tf.keras.layers.Normalization(
    input_shape=(1,),
    axis=None
)
all_normalizer.adapt(X_train_all.reshape(-1))

all_nn_model =  tf.keras.Sequential([
    all_normalizer,
    tf.keras.layers.Dense(len(df.columns[1:])),
    tf.keras.layers.Dense(1),
])

all_nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_squared_error')

# print(X_train_all)
# print('@@@')
# print(X_train_all.reshape(-1))
# print('##')

history = all_nn_model.fit( X_train_all.reshape(-1), y_train_all,
    verbose=0,
    epochs=1000,
    validation_data=(X_valid_all, y_valid_all)
)


plt.scatter(X_train_all, y_train_all, label='Data', color='blue')
x = tf.linspace(-20, 40, 100)
plt.plot(x, all_nn_model.predict(np.array(x).reshape(-1, 1)), label='Fit', color='red', linewidth=3)
plt.legend()
plt.title('bikes vs neural linear regression')
plt.ylabel('number of bikes')
plt.xlabel('all')
plt.show()
