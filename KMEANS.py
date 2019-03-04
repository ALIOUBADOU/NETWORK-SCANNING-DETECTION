#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('first_data.csv')
#%%
#%%
dataset=dataset[dataset['class'] != 'unknown']
#%%
#%%
X=dataset.iloc[:,[1,2,7,10,12]]
#%%
#%%
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_1 = LabelEncoder()
X['Proto'] = labelencoder_1.fit_transform(X['Proto'])
labelencoder_2 = LabelEncoder()
X['class'] = labelencoder_2.fit_transform(X['class'])
labelencoder_3 = LabelEncoder()
X['Flags'] = labelencoder_3.fit_transform(X['Flags'])
onehotencoder = OneHotEncoder(categorical_features = [1,3])
X= onehotencoder.fit_transform(X).toarray()
#%%
#%%
# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#%%
#%%# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
#%%
#%%
# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
#%%