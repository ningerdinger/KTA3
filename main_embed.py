import numpy as np
import pandas as pd
import cv2
from pathlib import Path  # import Path from pathlib module
from embed import embed_face_net


def embed(FACES_FOLDER_TRAINING, OUTPUT_FOLDER_RESULTS, RESULTS_NAME):
    """
    Processes face images from a specified folder, computes their embeddings using a neural network, 
    and saves the results to a CSV file.

    Parameters:
        FACES_FOLDER_TRAINING (str): Path to the folder containing face images.
        OUTPUT_FOLDER_RESULTS (str): Path to the folder where the results CSV will be saved.
        RESULTS_NAME (str): Name of the CSV file to save the computed embeddings.

    Returns:
        None
    """
    directory = Path(FACES_FOLDER_TRAINING)  # Convert the training folder path to a Path object for easier iteration.
    vector_embedding = dict()  # Initialize a dictionary to store embeddings keyed by the image filenames.

    for file in directory.iterdir():  # Iterate over all files in the face folder.
        if file.is_file():  # Process only if the item is a file.
            file_name = file.name  # Extract the file name for reference.
            print(file_name)  # Debugging: Print the name of the current file being processed.

            path = FACES_FOLDER_TRAINING + file_name  # Build the complete file path to the image.
            img = cv2.imread(path)  # Load the image into memory using OpenCV.

            vector = embed_face_net(img)  # Compute the embedding for the face using the embed_face_net model.
            vector_embedding[file.name] = vector.detach().numpy().flatten()  # Convert the embedding to a NumPy array, flatten it, and store in the dictionary.

    embedding_df = pd.DataFrame(vector_embedding)  # Convert the dictionary of embeddings into a Pandas DataFrame.
    print(embedding_df.shape)  # Debugging: Print the DataFrame dimensions (rows, columns) for verification.

    embedding_df.to_csv(OUTPUT_FOLDER_RESULTS + RESULTS_NAME, index=False)  # Save the DataFrame as a CSV file in the output folder.


