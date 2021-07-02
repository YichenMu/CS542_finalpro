import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


# to load data into a dataframe
def load_dataset(path):
    r_data = pd.read_csv(path, keep_default_na=False)
    return r_data


# remove the unnecessary cols
def remove_cols(dataset, cols_to_clean):
    r_data = dataset.drop([cols_to_clean], axis=1)
    return r_data


def engineer_features(dataset, cols, col):
    dataset[cols] = dataset[cols].div(dataset[col].values, axis=0)
    return dataset


def normalize(dataset, cols_to_normalize):
    for col in cols_to_normalize:
        dataset[col] = (dataset[col] - dataset[col].mean()) / dataset[col].std()
    return


def plot_corr_matrix(data, cols):
    corr = data[cols].corr()
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True, annot=True, fmt='.3f'
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );
    plt.show()


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
features = ['pts', 'oreb', 'dreb', 'reb', 'asts', 'stl', 'blk', 'turnover', 'pf', 'fga', 'fgm', 'fta', 'tpa', 'tpm']
data = remove_cols(load_dataset("player_regular_season_career.txt"), "leag")
# to remove the rows with 'minutes' as 0
clear_data = data.drop(data[(data['minutes'] == 0)].index)
clear_data[features] = clear_data[features].div(clear_data["minutes"].values, axis=0)
clear_data['mpg'] = clear_data['minutes'] / clear_data['gp']
clear_data = clear_data.drop(clear_data[(clear_data['mpg'] < 24)].index).reset_index(drop=True)
normalize(clear_data, features)

# x = clear_data[features].values
# pca = PCA(n_components=2)
# principalComponents = pca.fit_transform(x)
# principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
# # print(principalDf)
# finalDf = pd.concat([clear_data[['ilkid', 'firstname', 'lastname']], principalDf], axis=1)
# X = finalDf[['principal component 1', 'principal component 2']].values


def dbscan(data):
    outliers = data[(data['minutes'] == 0)];
    data = data.drop(data[(data['minutes'] == 0)].index)
    data[features] = data[features].div(data["minutes"].values, axis=0)
    data = data.reset_index(drop=True)
    normalize(data, features)
    # PCA part
    x = data[features].values
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([data[['ilkid', 'firstname', 'lastname']], principalDf], axis=1)
    # DBSCAN
    X = finalDf[['principal component 1', 'principal component 2']].values

    # # silhouette score
    # # Defining the list of hyperparameters to try
    # eps_list = np.arange(start=0.1, stop=0.9, step=0.01)
    # min_sample_list = np.arange(start=2, stop=10, step=1)
    # # Creating empty data frame to store the silhouette scores for each trials
    # silhouette_scores_data = pd.DataFrame()
    # for eps_trial in eps_list:
    #     for min_sample_trial in min_sample_list:
    #         # Generating DBSAN clusters
    #         db = DBSCAN(eps=eps_trial, min_samples=min_sample_trial)
    #         if (len(np.unique(db.fit_predict(X))) > 1):
    #             sil_score = silhouette_score(X, db.fit_predict(X))
    #         else:
    #             continue
    #         trial_parameters = "eps:" + str(eps_trial.round(1)) + " min_sample :" + str(min_sample_trial)
    #         silhouette_scores_data = silhouette_scores_data.append(
    #             pd.DataFrame(data=[[sil_score, trial_parameters]], columns=["score", "parameters"]))
    # # Finding out the best hyperparameters with highest Score
    # print(silhouette_scores_data.sort_values(by='score', ascending=False).head(1))

    db = DBSCAN(eps=.9, min_samples=6).fit(X)
    # Black removed and is used for noise instead.
    labels = db.labels_
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(2, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
        class_member_mask = (labels == k)
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14, alpha=0.6)
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6, alpha=0.6)
    plt.title('Estimated number of clusters: %d - player_playoffs' % n_clusters_)
    plt.show()
    finalDf['pred_dbscan'] = labels
    print(finalDf)
    data['pred_dbscan'] = labels
    print(data)
    print("\n")
    print(outliers)
    new_outliers = data[(data['pred_dbscan'] == -1)].drop(['pred_dbscan'], axis=1);
    print(new_outliers)
    final_outliers = pd.concat([outliers, new_outliers], axis=0)
    print(final_outliers)

    # outliers to export to csv
    df = pd.DataFrame(final_outliers)
    df.to_csv('outliers_player_playoffs.txt', index=False)
    return

# data_reg_season = remove_cols(load_dataset("player_regular_season_career.txt"), "leag")
# dbscan(data_reg_season)
data_reg_season = remove_cols(load_dataset("player_playoffs_career.txt"), "leag")
dbscan(data_reg_season)


# elbow method (=10
# X would be like finalDf[['principal component 1', 'principal component 2']].values
def elbow(X):
    wcss = []
    for i in range(1,30):
        kmeans = KMeans(n_clusters=i, max_iter=300, n_init=10, init='k-means++', random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1,30), wcss, 'c*-')
    plt.title('The Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()
    return

# # kmeans cluster and plot
# kmeans = KMeans(n_clusters=10, max_iter=300, n_init=10, init='k-means++', random_state=0)
# kmeans_res = kmeans.fit_predict(X)
# numSamples = len(X)
# mark = ['or', 'ob', 'og', 'ok', '^c', '+m', 'sy', 'dC0', '<C1', 'pC2']
# name = finalDf['firstname'].values + " " + finalDf['lastname'].values
# for i in range(numSamples):
#     #markIndex = int(clusterAssment[i, 0])
#     plt.plot(X[i][0], X[i][1], mark[kmeans.labels_[i]]) #mark[markIndex])
# for i, txt in enumerate(name):
#     plt.annotate(txt, (X[i][0], X[i][1]))
# plt.show()


# # to plot the correlation between features and principal component
# map= pd.DataFrame(pca.components_,columns=features)
# plt.figure(figsize=(12,6))
# sns.heatmap(map,cmap='twilight',annot=True)
# plt.show()


# Compute the correlation matrix
def corr_matrix(data):
    # corr = clear_data[features].corr()
    corr = data[features].corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );
    plt.show()
    return

