import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score, pairwise_distances_argmin_min
import os
import plots


def find_best_number_of_clusters(results_csv, min_clusters=2, max_clusters=15):
    """
    Identifies the best number of clusters for KMeans clustering by evaluating silhouette scores 
    and calinski-harabasz scores over a range of cluster counts. It also generates plots to visualize
    the scores.

    Parameters:
        results_csv (str): Path to the CSV file containing embeddings (features for clustering).
        min_clusters (int): Minimum number of clusters to test (default is 2).
        max_clusters (int): Maximum number of clusters to test (default is 15).

    Returns:
        tuple: 
            - best_n_clusters_silhouette (int): Number of clusters with the highest silhouette score.
            - best_n_clusters_calinski (int): Number of clusters with the highest calinski-harabasz score.
    """
    embeddings_df = pd.read_csv(results_csv)  # Load embeddings from the CSV file into a Pandas DataFrame.
    embeddings = embeddings_df.T.values  # Transpose the DataFrame to convert embeddings into feature vectors.

    # Initialize variables to store the best scores and corresponding cluster counts.
    best_silhouette_score = -1  # Highest silhouette score observed so far.
    best_calinski_score = -1  # Highest calinski-harabasz score observed so far.
    best_n_clusters_silhouette = min_clusters  # Best number of clusters based on silhouette score.
    best_n_clusters_calinski = min_clusters  # Best number of clusters based on calinski-harabasz score.

    silhouette_scores = []  # List to store silhouette scores for each cluster count.
    calinski_scores = []  # List to store calinski-harabasz scores for each cluster count.

    # Ensure required directories exist for saving plots and k-means results.
    if not os.path.exists('Plots'):
        os.makedirs('Plots')  # Create the Plots folder if it does not exist.
    if not os.path.exists('KMEANS_OUTPUT'):
        os.makedirs('KMEANS_OUTPUT')  # Create the KMEANS_OUTPUT folder if it does not exist.

    # Loop over the specified range of cluster counts.
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)  # Perform k-means clustering.
        labels = kmeans.labels_  # Get cluster labels assigned to each data point.

        # Calculate evaluation metrics for the current clustering result.
        silhouette_avg = silhouette_score(embeddings, labels)  # Compute average silhouette score.
        calinski_avg = calinski_harabasz_score(embeddings, labels)  # Compute calinski-harabasz score.

        # Append scores to their respective lists for later plotting.
        silhouette_scores.append(silhouette_avg)
        calinski_scores.append(calinski_avg)

        # Update best silhouette score and its cluster count if the current score is higher.
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_n_clusters_silhouette = n_clusters

        # Update best calinski-harabasz score and its cluster count if the current score is higher.
        if calinski_avg > best_calinski_score:
            best_calinski_score = calinski_avg
            best_n_clusters_calinski = n_clusters

    # Plot the scores using a custom function and save the results.
    plots.plot_scores(min_clusters, max_clusters, silhouette_scores, calinski_scores, 
                      best_n_clusters_silhouette, best_n_clusters_calinski)

    # Return the best cluster counts based on silhouette and calinski-harabasz scores.
    return best_n_clusters_silhouette, best_n_clusters_calinski