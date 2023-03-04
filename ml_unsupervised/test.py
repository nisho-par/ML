import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


cols = ['area', 'perimeter', 'compactness', 'kernel_length', 'width_kernel', 'asymmetry_coefficient', 'kernel_groove']
df = pd.read_csv('ml_unsupervised/seeds_dataset.txt', names=cols, sep='\s+')


# supposing that kernel_groove column isnt known
for i in range(len(cols)-1):
    for j in range(i+1, len(cols)-1):
        x_label = cols[i]
        y_label = cols[j]
        # sns.scatterplot(x=x_label, y=y_label, data=df, hue='kernel_groove')
        # plt.show()
        
# clustering
from sklearn.cluster import KMeans

x = 'perimeter'
y = 'asymmetry_coefficient'
X = df[[x, y]].values

kmeans = KMeans(n_clusters=3).fit(X)
clusters = kmeans.labels_

# print('clusters ', clusters)

kv_vals = df['kernel_groove'].values
# print('k means ',kv_vals)

clusters_df = pd.DataFrame(np.hstack((X, clusters.reshape(-1, 1))), columns=[x, y, 'kernel_groove'])

# k means classes
sns.scatterplot(x=x, y=y, hue='kernel_groove', data=clusters_df)
plt.plot()
plt.show()

# original classes
sns.scatterplot(x=x, y=y, hue='kernel_groove', data=df)
plt.plot()
plt.show()

# higher dimenstions
X = df[cols[:-1]].values
kmeans = KMeans(n_clusters=3).fit(X)
clusters_df = pd.DataFrame(np.hstack((X, clusters.reshape(-1, 1))), columns=df.columns)
#k means classes
sns.scatterplot(x=x, y=y, hue='kernel_groove', data=clusters_df)
plt.plot()
plt.show()
# original classes
sns.scatterplot(x=x, y=y, hue='kernel_groove', data=df)
plt.plot()
plt.show()



# PCA 
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
transformed_x = pca.fit_transform(X)
print(X.shape, transformed_x.shape)
print(transformed_x[:5])
plt.scatter(transformed_x[:, 0], transformed_x[:, 1])
plt.show()
k_means_pca_df = pd.DataFrame(np.hstack((transformed_x, kmeans.labels_.reshape(-1,1))), columns=['pca1', 'pca2', 'kernel_groove'])
truth_pca_df = pd.DataFrame(np.hstack((transformed_x, df['kernel_groove'].values.reshape(-1,1))), columns=['pca1', 'pca2', 'kernel_groove'])
#k means classes
sns.scatterplot(x='pca1', y='pca2', hue='kernel_groove', data=k_means_pca_df)
plt.plot()
plt.show()
# original classes
sns.scatterplot(x='pca1', y='pca2',hue='kernel_groove', data=truth_pca_df)
plt.plot()
plt.show()

