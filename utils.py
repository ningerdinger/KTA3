import numpy as np
import cv2 as cv
import mediapipe as mp
import os

def extract_frames(video_file_path, video_name, sample_every):
  print('Start extracting frames')
  frame_list = []
  video_path = os.path.join(video_file_path, video_name)
  cap = cv.VideoCapture(video_path)
  if not cap.isOpened():
    print("Error: Cannot open video file.")
    return []
  i = 0
  captured = 0
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
    if i % sample_every == 0:
      frame_rgb = np.ascontiguousarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
      frame_list.append(mp_image)
      captured += 1
    i += 1
  cap.release()
  cv.destroyAllWindows()
  print(f"Successfully captured {captured} frames")
  return frame_list
  
def save_face_list(face_list_movie,faces_folder,movie,extension):
  saved_faces = 0
  for i, face in enumerate(face_list_movie):
      if face.shape[0] > 10 and face.shape[1] > 10:
          cv.imwrite(faces_folder + movie + '_' + str(saved_faces) + extension, face)
          saved_faces += 1
  return