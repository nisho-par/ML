import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv('ml_continuous/SeoulBikeData.csv', encoding='unicode_escape')

cols = df.columns

# drop columns
df = df.drop(['Date', 'Seasons', 'Holiday'], axis=1)
df['Functioning Day'] = (df['Functioning Day'] == 'Yes').astype(int)

# filter data by value of column
r = df.loc[df['Functioning Day']==1]
# print(r)


df = df[df['Hour']==12]
df = df.drop(['Hour'], axis=1)

# column 1 is y
for label in df.columns[1:]:
    plt.scatter(x=df[label], y=df['Rented Bike Count'])
    plt.title(label)
    plt.xlabel(label)
    plt.ylabel('bike count at Noon')
    # plt.show()

