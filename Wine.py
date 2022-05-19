# Importing Required Packages
import pandas as pd 
import numpy as np

# Loading Dataset
data = pd.read_csv("C:\\Users\\SHAJIUDDIN MOHAMMED\\Desktop\\wine.csv")
data.describe()

# Removing Unnecessary Columns
Data = data.drop(["Type"], axis = 1)

# Importing PCA and Scale for PCA Analysis from sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale 

# Normalizing the numerical data 
data_normal = scale(Data)
data_normal

pca = PCA(n_components = 6)
pca_values = pca.fit_transform(data_normal)

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var

pca.components_
pca.components_[0]
# Cumulative variance 

var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1

import matplotlib.pylab as plt
# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")

# PCA scores
pca_values

pca_data = pd.DataFrame(pca_values)
pca_data.columns = "comp0", "comp1", "comp2", "comp3", "comp4", "comp5"
final = pd.concat([data.Type, pca_data.iloc[:, 0:3]], axis = 1)

# Scatter diagram
plt.scatter(x = final.comp0, y = final.comp1)


###############################   HIERARCHICAL CLUSTERING   #######################################

# Importing Cdist and Kmeans for Hierarchical clustering from Sklearn
from scipy.spatial.distance import cdist 
from sklearn.cluster import KMeans

# Loading Dataset
data = pd.read_csv("C:\\Users\\SHAJIUDDIN MOHAMMED\\Desktop\\wine.csv")
data.describe()

# Removing Unnecessary Columns
Data = data.drop(["Type"], axis = 1)

############Normalizing the data#################
def norm_func(i):
    x=(i-i.min()/i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm=norm_func(Data.iloc[:,0:])
df_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
z=linkage(df_norm,method="complete",metric="euclidean")

# Dendrogram
plt.figure(figsize=(25,50));plt.title("Dendogram");plt.xlabel("Index");plt.ylabel("Distance")
sch.dendrogram(
    z,
    leaf_rotation=0.,
    leaf_font_size=6.,
)
plt.show()


#################NOw applying agglomerative clustering########
from sklearn.cluster import AgglomerativeClustering
h_labels=AgglomerativeClustering(n_clusters=14,affinity="euclidean",linkage="complete").fit(df_norm)
clusters_labels=pd.Series(h_labels.labels_)

data["Clusters"]=clusters_labels
df_final=data.iloc[:,[0,14,1,2,3,4,5,6,7,8,9,10,11,12,13]]
df_final.to_csv("wine_py.csv")


####################################   K-MEANS CLUSTERING   #######################################

# Importing Dataset
data = pd.read_csv("C:\\Users\\SHAJIUDDIN MOHAMMED\\Desktop\\wine.csv")
data.describe()

# Removing Unnecessary Columns
Data = data.drop(["Type"], axis = 1)

############Normalizing the data#################
def norm_func(i):
    x=(i-i.min()/i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm=norm_func(Data.iloc[:,0:])
df_norm.describe()

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
Data['clust'] = mb # creating a  new column and assigning it to new column 

Data.head()
df_norm.head()

wine = Data.iloc[:,[0,13,1,2,3,4,5,6,7,8,9,10,11,12]]
wine.head()

wine.iloc[:, 1:14].groupby(Data.clust).mean()

Data.to_csv("Kmeans_Wine.csv", encoding = "utf-8")

import os
os.getcwd()

