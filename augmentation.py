import os
import cv2
import numpy as np
from tqdm import tqdm

def adjust_brightness_contrast(img, alpha=1.2, beta=30):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# Function: Add Gaussian Noise
def add_gaussian_noise(img):
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

# Function: Apply Motion Blur
def apply_blur(img, ksize=10):
    # Ensure ksize is a positive odd integer
    if ksize <= 0 or ksize % 2 == 0:
        raise ValueError("Kernel size (ksize) must be a positive odd integer.")
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

# Function: Rotate Image
def rotate_image(img, angle=45):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h))

# Function: Apply Cutout (Random Black Box)
def apply_cutout(img, size=80):
    newimg = img
    h, w = newimg.shape[:2]
    x1, y1 = np.random.randint(0, w - size), np.random.randint(0, h - size)
    newimg[y1:y1 + size, x1:x1 + size] = 0  # Apply black box
    return newimg

def distort_image(image, distortion_level):
    h, w = image.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([
        [0, 0 + distortion_level],
        [w, 0 - distortion_level],
        [w, h + distortion_level],
        [0, h - distortion_level]
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    distorted_image = cv2.warpPerspective(image, M, (w, h))
    return distorted_image



def process_images_with_augmentations(input_dir, output_dir):
    # Define augmentation functions and their names
    augmentations = {
        "horizontal_flip": lambda img: cv2.flip(img, 1),
        "brightness_contrast": lambda img: adjust_brightness_contrast(img, alpha=3, beta=20),
        "gaussian_noise": add_gaussian_noise,
        #"motion_blur": apply_blur,
        "rotation": lambda img: rotate_image(img, angle=45),
        "distortion": lambda img: distort_image(img, distortion_level=60),
        "cutout": apply_cutout
    }

    # Ensure output directories for each augmentation exist
    for aug_name in augmentations.keys():
        aug_output_path = os.path.join(output_dir, aug_name)
        os.makedirs(aug_output_path, exist_ok=True)

    # Process each image in the input directory
    for img_name in tqdm(os.listdir(input_dir), desc="Processing images"):
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        # Apply each augmentation and save the result
        for aug_name, aug_function in augmentations.items():
            aug_img = aug_function(image)
            aug_img_name = f"{os.path.splitext(img_name)[0]}_{aug_name}.jpg"
            aug_img_path = os.path.join(output_dir, aug_name, aug_img_name)
            cv2.imwrite(aug_img_path, aug_img)

