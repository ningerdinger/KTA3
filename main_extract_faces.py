import numpy as np
import cv2 as cv
from utils import save_face_list, extract_frames
from face_extraction import process_image, check_face
from facenet_pytorch import MTCNN
import torch

MOVIE_FOLDER = 'D:\\KTAI\\assignments\\3\\movies\\'
FRAME_FOLDER = 'D:\\KTAI\\assignments\\3\\output_images\\'
FACES_FOLDER_TRAINING = 'D:\\KTAI\\assignments\\3\\repo\\face_folder\\'
FACES_FOLDER_TEST = ''
MOVIE_TRAINING_LIST = ['New Kids ABC','New Kids Fussballspiel','New Kids Turbo_ Tankstation']
MOVIE_TEST_LIST = ['New Kids Nitro, _Peter lemonade!_ 720']

output_extension= '.png'
input_extension = '.mp4'
samples_per_second = 10         #FPS rate is assumed 25
padding_x = 10
padding_y = 10
min_confidence = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

for movie in MOVIE_TRAINING_LIST:
  print(movie+input_extension)
  frame_list = extract_frames(MOVIE_FOLDER,movie+input_extension,samples_per_second)
  face_list_movie = []
  for frame in frame_list:
    face_list_frame = process_image(frame,padding_x,padding_y,min_confidence)
    for face in face_list_frame:
      if check_face(face,mtcnn):
        face_list_movie.append(face)
  print('saving')
  save_face_list(face_list_movie,FACES_FOLDER_TRAINING,movie,output_extension)
