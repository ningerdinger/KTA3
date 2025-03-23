import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms


def embed_face_net(image):
  model = InceptionResnetV1(pretrained='vggface2').eval()
  img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((160, 160)),transforms.ToTensor()])
  img_tensor = transform(img).unsqueeze(0)
  with torch.no_grad():
    embedding = model(img_tensor)
  return embedding
  
