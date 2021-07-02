# CS542_prj
This is our final project of course CS542. The data we used can be publicly found at http://www.cs.cmu.edu/~ggordon/10601/projects.html

## Description
The goal of this project is to analyze the NBA Statistics in some machine learning ways. By saying analyzing, we tried to predict NBA results of 2003 based on related statistics of players, coaches and results of season games of previous years. Also, we tried to figure out the outliers of players using the players’ statistics. Besides, for specially exercising unsupervised learning technique, we clustered the players with statistics of their performance during season games. For the prediction part, we used techniques KNN, SVM and Decision Tree. We applied the DBSCAN clustering method to find the outliers. Then with the very common Kmeans++ we cluster the players to explore this famous unsupervised learning technique.

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

## Authors
Yichen Mu, Fangxu Zhou, Jingxuan Guo, Fu Hao

## Acknowledgments
We'd like to express gratitude to our professor Peter Chin and our two teaching fellows Peilun Dai and Andrew Wood. Without their help, we chouldn't have finished this project.
