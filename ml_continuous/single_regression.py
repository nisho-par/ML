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

# after seen the scatter plot drop the columns that doesnt show any kind of linearity or information
df = df.drop(['Wind speed (m/s)', 'Visibility (10m)', 'Functioning Day'], axis=1)

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

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

_, X_train_temp, y_train_temp = get_xy(train, 'Rented Bike Count', x_labels=['Temperature(°C)'])
_, X_valid_temp, y_valid_temp = get_xy(valid, 'Rented Bike Count', x_labels=['Temperature(°C)'])
_, X_test_temp, y_test_temp = get_xy(test, 'Rented Bike Count', x_labels=['Temperature(°C)'])

# print(X_train_temp,y_train_temp)

temp_reg = LinearRegression()
temp_reg.fit(X_train_temp, y_train_temp)


# print(temp_reg.coef_, temp_reg.intercept_, temp_reg.score(X_test_temp, y_test_temp))

plt.scatter(X_train_temp, y_train_temp, label='Data', color='blue')
x = tf.linspace(-20, 40, 100)
plt.plot(x, temp_reg.predict(np.array(x).reshape(-1, 1)), label='Fit', color='red', linewidth=3)
plt.legend()
plt.title('bikes vs temp')
plt.ylabel('number of bikes')
plt.xlabel('temp')
plt.show()
# to fix, because i see more data and not only temperature 

