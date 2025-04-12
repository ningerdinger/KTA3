import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms


def embed_face_net(image):
    """
    Embeds a face from an input image using the pre-trained InceptionResnetV1 model.

    Parameters:
        image (numpy.ndarray): Input image in BGR format.
    Returns:
        torch.Tensor: A 1x512-dimensional embedding vector.
    """
    model = InceptionResnetV1(pretrained='vggface2').eval()  # Load pre-trained model.
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB.
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((160, 160)), transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)  # Resize, normalize, add batch dimension.
    with torch.no_grad():
        embedding = model(img_tensor)  # Compute embedding in evaluation mode.
    return embedding  # Return embedding vector.
