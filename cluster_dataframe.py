
from utils import save_face_list, extract_frames
from face_extraction import process_image, check_face
from facenet_pytorch import MTCNN
import torch
from embed import embed_face_net
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min
import plots
import numpy as np
import shutil
import os

def process_videos_to_dataframe(movie_test_list, movie_folder, kmeans, samples_per_second=10, padding_x=10,
                                padding_y=10, min_confidence=0.5, threshold_distance_85=0.85):
    cluster_presence = pd.DataFrame()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    for movie in movie_test_list:
        print(f"Processing {movie + '.mp4'}")
        frame_list = extract_frames(movie_folder, movie + '.mp4', samples_per_second)
        cluster_frame_data = []

        for frame_idx, frame in enumerate(frame_list):
            face_list_frame = process_image(frame, padding_x, padding_y, min_confidence)
            frame_clusters = []

            for face in face_list_frame:
                if check_face(face, mtcnn):
                    embedding = embed_face_net(face).flatten()
                    embedding_reshaped = embedding.reshape(1, -1)

                    if pairwise_distances_argmin_min(embedding_reshaped, kmeans.cluster_centers_)[
                        1] > threshold_distance_85:
                        cluster = -1
                    else:
                        cluster = int(kmeans.predict(embedding_reshaped)[0])
                    frame_clusters.append(cluster)

            cluster_frame_data.append(frame_clusters)

        for frame_idx, clusters in enumerate(cluster_frame_data):
            for cluster in clusters:
                cluster_presence.loc[cluster, frame_idx] = 1

    cluster_presence.fillna(0, inplace=True)

    return cluster_presence


def process_test_embeddings(test_embeddings, test_embeddings_df, kmeans, output_folder, face_folder_test,results_output_path = 'image_to_cluster_results.csv'):
    # Predict clusters and calculate distances
    predicted_clusters = kmeans.predict(test_embeddings)
    distances_test = pairwise_distances_argmin_min(test_embeddings, kmeans.cluster_centers_)[1]
    threshold_distance_85 = np.percentile(distances_test, 85)

    # Plot histogram
    plots.plot_histplot_percentile(distances_test, threshold_distance_85)

    # Create image-to-cluster mapping
    image_cluster_results = []
    for i, cluster in enumerate(predicted_clusters):
        if distances_test[i] > threshold_distance_85:
            cluster = -1
        image_cluster_results.append({
            "Image": test_embeddings_df.columns[i],
            "Cluster": cluster
        })
        print(f"Image {test_embeddings_df.columns[i]} belongs to Cluster {cluster}")

    # Save image-to-cluster mapping
    results_df = pd.DataFrame(image_cluster_results)
    results_output_path = results_output_path
    results_df.to_csv(results_output_path, index=False)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Organize images into cluster folders
    for _, row in results_df.iterrows():
        image_name = row['Image']
        cluster = row['Cluster']
        cluster_folder = os.path.join(output_folder, f'Cluster_{cluster}')
        os.makedirs(cluster_folder, exist_ok=True)

        source_path = os.path.join(face_folder_test, image_name)
        destination_path = os.path.join(cluster_folder, image_name)

        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)

    return results_df