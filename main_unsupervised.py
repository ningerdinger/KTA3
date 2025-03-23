import pandas as pd
from sklearn.cluster import KMeans
import os
import shutil

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

RESULTS_CSV = 'D:\\KTAI\\assignments\\3\\results\\\\first_result.csv'
FACES_FOLDER = 'D:\\KTAI\\assignments\\3\\face_folder\\'
OUTPUT_BASE_FOLDER = 'clusters'

separate_images_by_clusters(RESULTS_CSV, FACES_FOLDER, OUTPUT_BASE_FOLDER, n_clusters=8)