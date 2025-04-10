import numpy as np
import pandas as pd
import cv2
from pathlib import Path  # import Path from pathlib module
from embed import embed_face_net


def embed(FACES_FOLDER_TRAINING,OUTPUT_FOLDER_RESULTS,RESULTS_NAME):
  directory = Path(FACES_FOLDER_TRAINING)
  vector_embedding = dict()
  for file in directory.iterdir():
    if file.is_file():
      file_name = file.name
      print(file_name)
      path = FACES_FOLDER_TRAINING + file_name
      img = cv2.imread(path)
      vector = embed_face_net(img)
      vector_np = vector.detach().numpy()
      vector_embedding[file.name] = vector_np.flatten()

  embedding_df = pd.DataFrame(vector_embedding)
  print(embedding_df.shape)

  embedding_df.to_csv(OUTPUT_FOLDER_RESULTS+RESULTS_NAME, index=False)


