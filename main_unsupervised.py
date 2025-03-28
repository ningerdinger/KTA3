import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
import os
import shutil
import joblib


def find_best_number_of_clusters(results_csv, min_clusters=2, max_clusters=10):
    embeddings_df = pd.read_csv(results_csv)
    embeddings = embeddings_df.T.values
    best_silhouette_score = -1
    best_calinski_score = -1
    best_n_clusters_silhouette = min_clusters
    best_n_clusters_calinski = min_clusters
    silhouette_scores = []
    calinski_scores = []


    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
        labels = kmeans.labels_
        joblib.dump(kmeans, os.path.join(OUTPUT_BASE_FOLDER, 'kmeans_model.pkl'))
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

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o', label='Silhouette Score')
    plt.axvline(x=best_n_clusters_silhouette, color='r', linestyle='--', label=f'Best Clustering ({best_n_clusters_silhouette})')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.legend()
    plt.title('Silhouette score vs number of clusters')

    plt.subplot(1, 2, 2)
    plt.plot(range(min_clusters, max_clusters + 1), calinski_scores, marker='o', label='Calinski-Harabasz Score')
    plt.axvline(x=best_n_clusters_calinski, color='r', linestyle='--', label=f'Best Clustering ({best_n_clusters_calinski})')
    plt.xlabel('Number of clusters')
    plt.ylabel('Calinski-Harabasz score')
    plt.legend()
    plt.title('Calinski-Harabasz score vs number of clusters')


    plt.tight_layout()

    if not os.path.exists('Plots'):
        os.makedirs('Plots')
    plt.savefig('Plots/clustering_scores.png')
    plt.show()

    return best_n_clusters_silhouette, best_n_clusters_calinski

def separate_images_by_clusters(results_csv, faces_folder, output_base_folder, n_clusters=2):
    embeddings_df = pd.read_csv(results_csv)
    embeddings = embeddings_df.T.values

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    labels = kmeans.labels_

    for cluster in range(n_clusters):
        cluster_folder = os.path.join(output_base_folder, f'cluster_{cluster}')
        os.makedirs(cluster_folder, exist_ok=True)

    for i, file_name in enumerate(embeddings_df.columns):
        src_path = os.path.join(faces_folder, file_name)
        dst_path = os.path.join(output_base_folder, f'cluster_{labels[i]}', file_name)
        shutil.copy(src_path, dst_path)

RESULTS_CSV = 'D:\\KTAI\\assignments\\3\\repo\\results\\second_result.csv'
FACES_FOLDER = 'D:\KTAI\\assignments\\3\\repo\\face_folder\\'
OUTPUT_BASE_FOLDER = 'KMEANS_OUTPUT'

best_clusters_silhouette, best_clusters_calinski = find_best_number_of_clusters(RESULTS_CSV)
separate_images_by_clusters(RESULTS_CSV, FACES_FOLDER, OUTPUT_BASE_FOLDER, n_clusters=best_clusters_silhouette)

'''print(f"Best number of clusters (Silhouette): {best_clusters_silhouette}")
print(f"Best number of clusters (Calinski-Harabasz): {best_clusters_calinski}")'''


''' Voor het laden van kmeans model
import joblib
kmeans = joblib.load(os.path.join(output_base_folder, 'kmeans_model.pkl'))'''