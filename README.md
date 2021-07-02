# CS542_prj
This is our final project of course CS542. The data we used can be publicly found at http://www.cs.cmu.edu/~ggordon/10601/projects.html

## Description
The goal of this project is to analyze the NBA Statistics in some machine learning ways. By saying analyzing, we tried to predict NBA results of 2003 based on related statistics of players, coaches and results of season games of previous years. Also, we tried to figure out the outliers of players using the players’ statistics. Besides, because our data lack the label columns congenitally, in order to add the labels to training and testing data and specially exercise unsupervised learning technique, we firstly clustered the players with statistics of their performance during season games. Then for the prediction part, we used techniques KNN, SVM, Decision Tree and Neural Networks. We also applied the DBSCAN clustering method to find the outliers. Then with the very common Kmeans++ we cluster the players to explore this famous unsupervised learning technique.

## Getting Started

### Language
Python

### Dependencies
import math  
import numpy as np  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt  
import sklearn.tree as sk_tree  
from sklearn import preprocessing  
from sklearn.svm import SVC  
from sklearn.cluster import KMeans  
from sklearn.cluster import DBSCAN  
from sklearn.metrics import mean_squared_error  
from sklearn.metrics import silhouette_score  
from sklearn.pipeline import Pipeline  
from sklearn.pipeline import make_pipeline  
from sklearn.decomposition import PCA  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.neural_network import MLPClassifier  

### Working procedure
Because there are some congenital defects in our original datasets, including player, team,and coaches datasets. They all lack the label data. With that problem, we couldn’t implement Supervised Learning algorithm on our dataset. So after preprocessing the data, we need to use K-means++ clustering algorithm to cluster the data and label them according to the sum of their feature values. In this way, there will be label data in our training dataset. After finishing these works, we can then use the supervised learning algorithms like knn on the training data later.  

So our working process is like this:  

Firstly, we select three data files, player regular season, coaches season and team season, and merge them into one main dataset.  Then we filter the dataset by year. We divide the dataset which we got from the last step into two parts. One is Training data and the other is testing data.  Then, in order to make our predicting models more accurate and our features more representative, we calculate and create some new feature columns based on the original feature columns. The formulas we used is in the page 33 of our project slides.  After that, we firstly use minmaxscaler to pre-process the data. Then we define two functions which are elbow function and create_label function. In elbow function, we used kmeans++ algorithm and matplotlib package to plot images which are related to the optimal number for the clusters. We can see the images in files we uploaded. According to the images, we can implement create_label function to calculate the label for every row of data in both training dataset and testing dataset. Then we can get the datasets with labels. 

As for the supervised learning part, we used the make_pipeline method to construct a pipeline to deal with the data, the make_pipeline we used contains minmaxscaler, Principal Component Analysis and different learning algorithms. And as for the learning algorithms, we tried K-NearestNeighbor, Support vector machine, decision tree and neural network. As for the detailed information about parameters, we listed them in the page 40 of our presentation slides.  Then comparing the accuracies of different learning models, we can tell the Neural Network will be the most optimal choice to fit the machine learning model and use it to predict the winning rate of teams in 2003!






## Authors
Yichen Mu, Fangxu Zhou, Jingxuan Guo, Fu Hao

## Acknowledgments
We'd like to express gratitude to our professor Peter Chin and our two teaching fellows Peilun Dai and Andrew Wood. Without their help, we chouldn't have finished this project.
