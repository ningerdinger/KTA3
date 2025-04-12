import numpy as np
import cv2 as cv
from utils import save_face_list, extract_frames
from face_extraction import process_image, check_face
from facenet_pytorch import MTCNN
import torch

def process_movies(movie_list, movie_folder, faces_folder, input_extension, output_extension, samples_per_second, padding_x, padding_y, min_confidence):
    """
    Processes a list of movie files to extract face images, then saves them using the specified parameters.

    Parameters:
        movie_list (list): List of movie names to process.
        movie_folder (str): Path to the folder containing movie files.
        faces_folder (str): Path to the folder where face images will be saved.
        input_extension (str): File extension for input movie files (e.g., '.mp4').
        output_extension (str): File extension for output face images (e.g., '.png').
        samples_per_second (int): Number of frames to extract from each second of the video.
        padding_x (int): Horizontal padding around detected faces.
        padding_y (int): Vertical padding around detected faces.
        min_confidence (float): Minimum confidence threshold for face detection.

    Returns:
        None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, otherwise fallback to CPU.
    mtcnn = MTCNN(keep_all=True, device=device)  # Initialize the MTCNN face detection model on the appropriate device.

    for movie in movie_list:  # Loop through each movie in the list.
        print(f"Processing: {movie + input_extension}")  # Debugging: Print the name of the movie being processed.
        frame_list = extract_frames(movie_folder, movie + input_extension, samples_per_second)  # Extract frames from the video.

        face_list_movie = []  # List to store detected faces for the current movie.
        for frame in frame_list:  # Loop through all extracted frames.
            face_list_frame = process_image(frame, padding_x, padding_y, min_confidence)  # Detect faces in the frame.

            for face in face_list_frame:  # Loop through detected faces.
                if check_face(face, mtcnn):  # Verify the face using MTCNN before adding it to the list.
                    face_list_movie.append(face)  # Append the valid face to the list.

        print('Saving faces...')  # Debugging: Indicate the start of the saving process.
        save_face_list(face_list_movie, faces_folder, movie, output_extension)  # Save the detected faces to the specified folder.
