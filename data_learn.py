#-*- coding=utf-8 -*-
#@File:data_learn.py
#@Software:PyCharm


import sklearn.tree as sk_tree
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
import math
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier


def elbowTr(x):
    wcss = []
    for i in range(1,30):
        kmeanspp = KMeans(n_clusters=i, max_iter=300, n_init=10, init='k-means++', random_state=0)
        kmeanspp.fit(x)
        wcss.append(kmeanspp.inertia_)
    plt.plot(range(1,30), wcss, 'c*-')
    plt.title('The Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.savefig('img_traindata_elbow.png')
    plt.show()
    return

def elbowTe(x):
    wcss = []
    for i in range(1,30):
        kmeanspp = KMeans(n_clusters=i, max_iter=300, n_init=10, init='k-means++', random_state=0)
        kmeanspp.fit(x)
        wcss.append(kmeanspp.inertia_)
    plt.plot(range(1,30), wcss, 'c*-')
    plt.title('The Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.savefig('img_testdata_elbow.png')
    plt.show()
    return


def create_labels_Tr(x):
    for i in range(len(labels_Tr)):
        if i!=(len(labels_Tr)-1):
            if x.sum()<=labels_Tr[i] and x.sum()>=labels_Tr[i+1]:
                if abs(x.sum()-labels_Tr[i])<=abs(x.sum()-labels_Tr[i+1]):
                    return i
                else:
                    continue
            else:
                continue
        else:
            return len(labels_Tr)-1

def create_labels_Te(x):
    for i in range(len(labels_Te)):
        if i!=(len(labels_Te)-1):
            if x.sum()<=labels_Te[i] and x.sum()>=labels_Te[i+1]:
                if abs(x.sum()-labels_Te[i])<=abs(x.sum()-labels_Te[i+1]):
                    return i
                else:
                    continue
            else:
                continue
        else:
            return len(labels_Te)-1

np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_columns', None)

pd.set_option('display.width', 1000)
prepdata=pd.read_csv('databasebasketball2.0/Processed_final_data.txt',sep=',')
testdata=pd.read_csv('databasebasketball2.0/Processed_test_2003_data.txt',sep=',')
temp1=[0,35,39,40]
# Delete some non-numeric feature columns
prepdata.drop(prepdata.columns[temp1],axis=1,inplace=True)
testdata.drop(testdata.columns[temp1],axis=1,inplace=True)
# print(prepdata.shape)
scaler=preprocessing.MinMaxScaler()
finaldata=scaler.fit_transform(prepdata)
testdata_temp=scaler.fit_transform(testdata)

# elbowTr(finaldata)
# elbowTe(testdata)

# In order to add labels, we use Kmeans++ clustering algorithm to cluster the training data and testing data to find the clusters and centroids.
kmeanspp_cluster_Tr=KMeans(n_clusters=5,max_iter=10,n_init=10,init='k-means++').fit(finaldata)
kmeanspp_cluster_Te=KMeans(n_clusters=5,max_iter=10,n_init=10,init='k-means++').fit(testdata_temp)
kmeanspp_centroids_Tr=scaler.inverse_transform(kmeanspp_cluster_Tr.cluster_centers_)
kmeanspp_centroids_Te=scaler.inverse_transform(kmeanspp_cluster_Te.cluster_centers_)
# print(kmeanspp_cluster.cluster_centers_)
kmeanspp_centroids_Tr=list(kmeanspp_centroids_Tr)
kmeanspp_centroids_Te=list(kmeanspp_centroids_Te)
# print(kmeanspp_centroids)
labels_Tr=[]
labels_Te=[]
for i in range(len(kmeanspp_centroids_Tr)):
    labels_Tr.append(kmeanspp_centroids_Tr[i].sum())
for i in range(len(kmeanspp_centroids_Te)):
    labels_Te.append(kmeanspp_centroids_Te[i].sum())
# for i in range(len(kmeanspp_centroids)):
#     k=np.array(kmeanspp_centroids[i])
#     print(k==kmeanspp_centroids[i])
    # print(k.sum())

# Sort the sum of features in 5 centroids in descending order
labels_Tr=sorted(labels_Tr,reverse=True)
labels_Te=sorted(labels_Te,reverse=True)
# print(labels)
# Use create_labels functions to add labels to training data and testing data respectively.
prepdata['label']=prepdata.apply(lambda x: create_labels_Tr(x), axis=1)
testdata['label']=testdata.apply(lambda x: create_labels_Te(x), axis=1)
# print(prepdata)
# To split the training and testing dataset into features and their corresponding labels separately.
Y_train=prepdata['label']
Y_test=testdata['label']
X_train=prepdata.drop(['label'],axis=1)
X_test=testdata.drop(['label'],axis=1)
print(X_train)
# print('\n')
print(Y_train)
# print('\n')
print(X_test)
# print('\n')
print(Y_test)

# Try different machine learning models
pca=PCA(n_components=0.98,svd_solver='auto')
knn=KNeighborsClassifier(n_neighbors=9)
decisionT=sk_tree.DecisionTreeClassifier(criterion='entropy',max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features=None,max_leaf_nodes=None,min_impurity_decrease=0)
svm=SVC(kernel='rbf',class_weight='balanced',C=3.0)
nn=MLPClassifier(solver='sgd',activation='relu',alpha=1e-4,hidden_layer_sizes=(300,300,300),random_state=1,max_iter=200,learning_rate_init=0.001)
model=make_pipeline(scaler,pca,nn)
model.fit(X_train,Y_train)
Y_pre=model.predict(X_test)
print(Y_pre)
print("RMSE on testing set = ", math.sqrt(mean_squared_error(Y_test, Y_pre)))