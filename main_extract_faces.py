import numpy as np
import cv2 as cv
from utils import save_face_list, extract_frames
from face_extraction import process_image, check_face
from facenet_pytorch import MTCNN
import torch

def process_movies(movie_training_list, movie_folder, faces_folder_training, input_extension, output_extension, samples_per_second, padding_x, padding_y, min_confidence):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)

    for movie in movie_training_list:
        print(movie + input_extension)
        frame_list = extract_frames(movie_folder, movie + input_extension, samples_per_second)
        face_list_movie = []
        for frame in frame_list:
            face_list_frame = process_image(frame, padding_x, padding_y, min_confidence)
            for face in face_list_frame:
                if check_face(face, mtcnn):
                    face_list_movie.append(face)
        print('saving')
        save_face_list(face_list_movie, faces_folder_training, movie, output_extension)