import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


# create dataframe from csv
df = pd.read_csv('ml1/dataset.csv')

# returns first 5 instances
r = df.head()
# print(r)

# returns possibile values for a column
r = df['Target'].unique()
# print(r)

# create or replace column with values 1/0 based on a condition
df['Target'] = (df['Target']=='Dropout').astype(int)
# print(df)

# list of column names of dataframe
r = df.columns
# print(r)

# return dataframe without a column, note that the column isnt removed from the original dataframe
r = df.drop('Target', axis=1)
# print(r)

# dataframe without a column
df1 = df.loc[:, df.columns != 'Target']
# print(r)

# shuffles dataframe
r = df.sample(frac=1)
# print(r)


# show plot for each column,target_converted
for label in df1.columns[:-1]:
    plt.hist(df[df['Target'] == 1][label], color='red', label='college dropout', alpha=0.7, density=True)
    plt.hist(df[df['Target'] == 0][label], color='blue', label='college not dropout', alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel('probability')
    plt.xlabel(label)
    plt.legend()
    # plt.show()
    
# create train, validation, test datasets from shuffled dataframe, with composition of 60% 20% 20%  
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

# scale dataset relative to the mean/standard deviation of column
def scale_dataset(dataframe, oversample=False):
    
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values
    # print("pre scaling: ", X)
    
    scaler = StandardScaler()
    # take X and transform all values
    X = scaler.fit_transform(X)
    # print("post scaling: ", X)

    if oversample:
        # sampling to match to take same qta of y
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)
        
    # create data with X(scaled), y: horizontaly stack. need to reshape y because is monodimensional (differently from X), 
    data = np.hstack( (X, np.reshape(y, (-1, 1))))    
    # print("data", data)
    
    return data, X, y

    
data, X, y = scale_dataset(df)
# oversample dataset to fix problem of having a very large difference of result for classification: example if having 4000 of y1 and 1500 of y2. it creates a problem on ML. should match better
# print('pre oversampling')
# print(len(train[train['Target']==1]), len(train[train['Target']==0]))

# fix the previous problem balancing the number of results
train, X_train, y_train = scale_dataset(train, oversample=True)
# print('after oversampling')
# print(sum(y_train==1), sum(y_train==0))


valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)

