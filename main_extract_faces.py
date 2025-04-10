import numpy as np
import cv2 as cv
from utils import save_face_list, extract_frames
from face_extraction import process_image, check_face
from facenet_pytorch import MTCNN
import torch

def process_movies(movie_list, movie_folder, faces_folder, input_extension, output_extension, samples_per_second, padding_x, padding_y, min_confidence):
    """
    Processes movies to extract and save faces using the given parameters.

    Parameters:
        movie_list (list): List of movie names to process.
        movie_folder (str): Path to the folder containing movie files.
        faces_folder (str): Path to the folder where face images will be saved.
        input_extension (str): File extension of input movie files (e.g., '.mp4').
        output_extension (str): File extension of output face images (e.g., '.png').
        samples_per_second (int): Number of frames to extract per second.
        padding_x (int): Horizontal padding around detected faces.
        padding_y (int): Vertical padding around detected faces.
        min_confidence (float): Minimum confidence for face detection.

    Returns:
        None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)

    for movie in movie_list:
        print(f"Processing: {movie + input_extension}")
        frame_list = extract_frames(movie_folder, movie + input_extension, samples_per_second)
        face_list_movie = []
        
        for frame in frame_list:
            face_list_frame = process_image(frame, padding_x, padding_y, min_confidence)
            for face in face_list_frame:
                if check_face(face, mtcnn):
                    face_list_movie.append(face)
        
        print('Saving faces...')
        save_face_list(face_list_movie, faces_folder, movie, output_extension)
