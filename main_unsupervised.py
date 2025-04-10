import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score, pairwise_distances_argmin_min
import os
import plots


# find_best_number_of_clusters: function that takes a csv file with embeddings and finds the best number of clusters for KMeans clustering.
# params
# results_csv:       the csv file with the embeddings
# min_clusters:      the minimum number of clusters to test
# max_clusters:      the maximum number of clusters to test
# return:            the best number of clusters based on silhouette score and calinski-harabasz score

def find_best_number_of_clusters(results_csv, min_clusters=2, max_clusters=15):
    embeddings_df = pd.read_csv(results_csv)
    embeddings = embeddings_df.T.values
    best_silhouette_score = -1
    best_calinski_score = -1
    best_n_clusters_silhouette = min_clusters
    best_n_clusters_calinski = min_clusters
    silhouette_scores = []
    calinski_scores = []

    if not os.path.exists('Plots'):
        os.makedirs('Plots')
    if not os.path.exists('KMEANS_OUTPUT'):
        os.makedirs('KMEANS_OUTPUT')

    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(embeddings, labels)
        calinski_avg = calinski_harabasz_score(embeddings, labels)
        silhouette_scores.append(silhouette_avg)
        calinski_scores.append(calinski_avg)
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_n_clusters_silhouette = n_clusters

        if calinski_avg > best_calinski_score:
            best_calinski_score = calinski_avg
            best_n_clusters_calinski = n_clusters

    plots.plot_scores(min_clusters, max_clusters, silhouette_scores, calinski_scores, best_n_clusters_silhouette,
                best_n_clusters_calinski)

    return best_n_clusters_silhouette, best_n_clusters_calinski


