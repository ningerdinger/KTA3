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
def apply_blur(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

# Function: Rotate Image
def rotate_image(img, angle=15):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h))

# Function: Apply Cutout (Random Black Box)
def apply_cutout(img, size=30):
    h, w = img.shape[:2]
    x1, y1 = np.random.randint(0, w - size), np.random.randint(0, h - size)
    img[y1:y1 + size, x1:x1 + size] = 0  # Apply black box
    return img


# Function to Process Each Character Folder
def process_character_images(character, input_dir, output_dir, augmentations=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    character_path = os.path.join(input_dir, character)
    output_character_path = os.path.join(output_dir, character)

    if not os.path.isdir(character_path):
        return

    os.makedirs(output_character_path, exist_ok=True)

    for img_name in tqdm(os.listdir(character_path), desc=f"Processing {character}"):
        img_path = os.path.join(character_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        # Generate Augmented Images
        for i in range(augmentations):
            aug_img = image.copy()

            # Randomly apply transformations
            if np.random.rand() < 0.5:
                aug_img = cv2.flip(aug_img, 1)  # Horizontal Flip
            if np.random.rand() < 0.5:
                aug_img = adjust_brightness_contrast(aug_img, alpha=1.5, beta=20)
            if np.random.rand() < 0.3:
                aug_img = add_gaussian_noise(aug_img)
            if np.random.rand() < 0.3:
                aug_img = apply_blur(aug_img)
            if np.random.rand() < 0.5:
                aug_img = rotate_image(aug_img, angle=np.random.randint(-15, 15))
            if np.random.rand() < 0.3:
                aug_img = apply_cutout(aug_img)

            # Save Augmented Image
            aug_img_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg"
            aug_img_path = os.path.join(output_character_path, aug_img_name)
            cv2.imwrite(aug_img_path, aug_img)

