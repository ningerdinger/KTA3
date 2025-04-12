
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

def process_videos_to_dataframe(movie_test_list, movie_folder, kmeans, samples_per_second=10, padding_x=10, padding_y=10, min_confidence=0.5, threshold_distance_85=0.85):
    """
    Processes movie frames to associate face embeddings with KMeans clusters, storing the results in a DataFrame.

    Parameters:
        movie_test_list (list): List of movie names to process.
        movie_folder (str): Path to the folder containing movie files.
        kmeans (KMeans): Pre-trained KMeans clustering model.
        samples_per_second (int): Number of frames extracted per second from the video (default is 10).
        padding_x (int): Horizontal padding around detected faces (default is 10).
        padding_y (int): Vertical padding around detected faces (default is 10).
        min_confidence (float): Minimum confidence threshold for face detection (default is 0.5).
        threshold_distance_85 (float): Distance threshold for cluster assignment (default is 0.85).

    Returns:
        pd.DataFrame: DataFrame indicating cluster presence across video frames.
    """
    cluster_presence = pd.DataFrame()  # Initialize an empty DataFrame to track cluster presence.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available; fallback to CPU.
    mtcnn = MTCNN(keep_all=True, device=device)  # Initialize MTCNN face detection model.

    for movie in movie_test_list:  # Loop through movies to process.
        print(f"Processing {movie + '.mp4'}")  # Debugging: Print the name of the movie being processed.
        frame_list = extract_frames(movie_folder, movie + '.mp4', samples_per_second)  # Extract frames from the movie.
        cluster_frame_data = []  # Initialize list to store cluster data for each frame.

        for frame_idx, frame in enumerate(frame_list):  # Iterate over extracted frames.
            face_list_frame = process_image(frame, padding_x, padding_y, min_confidence)  # Detect faces in the frame.
            frame_clusters = []  # List to store clusters detected in the current frame.

            for face in face_list_frame:  # Process detected faces.
                if check_face(face, mtcnn):  # Verify the face using MTCNN.
                    embedding = embed_face_net(face).flatten()  # Compute embedding and flatten the result.
                    embedding_reshaped = embedding.reshape(1, -1)  # Reshape embedding for clustering.

                    # Determine cluster assignment based on threshold distance.
                    if pairwise_distances_argmin_min(embedding_reshaped, kmeans.cluster_centers_)[1] > threshold_distance_85:
                        cluster = -1  # Assign cluster -1 if distance exceeds the threshold.
                    else:
                        cluster = int(kmeans.predict(embedding_reshaped)[0])  # Predict the cluster index.

                    frame_clusters.append(cluster)  # Append cluster assignment.

            cluster_frame_data.append(frame_clusters)  # Store clusters for the frame.

        # Update DataFrame with cluster presence across frames.
        for frame_idx, clusters in enumerate(cluster_frame_data):
            for cluster in clusters:
                cluster_presence.loc[cluster, frame_idx] = 1  # Mark presence of cluster in the frame.

    cluster_presence.fillna(0, inplace=True)  # Fill empty cells with 0 to indicate absence.
    return cluster_presence  # Return the DataFrame.


def process_test_embeddings(test_embeddings, test_embeddings_df, kmeans, output_folder, threshold, face_folder_test, results_output_path='image_to_cluster_results.csv'):
    """
    Maps test embeddings to KMeans clusters, saves the mapping to a CSV file, and organizes images into cluster folders.

    Parameters:
        test_embeddings (np.ndarray): Array of test embeddings.
        test_embeddings_df (pd.DataFrame): DataFrame containing test embeddings.
        kmeans (KMeans): Pre-trained KMeans clustering model.
        output_folder (str): Path to the folder where cluster folders will be created.
        threshold (float): Distance threshold for cluster assignment.
        face_folder_test (str): Path to the folder containing test face images.
        results_output_path (str): Path to the CSV file where the results will be saved (default is 'image_to_cluster_results.csv').

    Returns:
        pd.DataFrame: DataFrame containing image-to-cluster mapping.
    """
    # Predict clusters and calculate distances for test embeddings.
    predicted_clusters = kmeans.predict(test_embeddings)  # Predict cluster indices for test embeddings.
    distances_test = pairwise_distances_argmin_min(test_embeddings, kmeans.cluster_centers_)[1]  # Compute distances to cluster centers.

    plots.plot_histplot_percentile(distances_test, threshold)  # Plot histogram of distances with percentile threshold.

    # Create image-to-cluster mapping.
    image_cluster_results = []
    for i, cluster in enumerate(predicted_clusters):
        if distances_test[i] > threshold:  # Assign cluster -1 if distance exceeds the threshold.
            cluster = -1
        image_cluster_results.append({
            "Image": test_embeddings_df.columns[i],  # Map image name to cluster.
            "Cluster": cluster
        })
        print(f"Image {test_embeddings_df.columns[i]} belongs to Cluster {cluster}")  # Debugging: Print image-to-cluster mapping.

    results_df = pd.DataFrame(image_cluster_results)  # Convert mapping to a DataFrame.
    results_df.to_csv(results_output_path, index=False)  # Save the mapping to a CSV file.

    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist.

    # Organize images into cluster folders.
    for _, row in results_df.iterrows():
        image_name = row['Image']  # Get the image name.
        cluster = row['Cluster']  # Get the cluster assignment.
        cluster_folder = os.path.join(output_folder, f'Cluster_{cluster}')  # Construct cluster folder path.
        os.makedirs(cluster_folder, exist_ok=True)  # Create the cluster folder if it doesn't exist.

        source_path = os.path.join(face_folder_test, image_name)  # Get the source path of the image.
        destination_path = os.path.join(cluster_folder, image_name)  # Construct destination path for the image.

        if os.path.exists(source_path):  # Check if the source image exists.
            shutil.copy(source_path, destination_path)  # Copy the image to the cluster folder.

    return results_df  # Return the DataFrame containing image-to-cluster mapping.
