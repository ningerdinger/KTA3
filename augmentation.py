import os
import cv2
import numpy as np
from tqdm import tqdm

def adjust_brightness_contrast(img, alpha=1.2, beta=30):
    """
    Adjusts the brightness and contrast of an image.
    
    Parameters:
        img (numpy.ndarray): Input image.
        alpha (float): Contrast control (default is 1.2). Higher values increase contrast.
        beta (int): Brightness control (default is 30). Higher values increase brightness.
        
    Returns:
        numpy.ndarray: Image with adjusted brightness and contrast.
    """
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def add_gaussian_noise(img):
    """
    Adds Gaussian noise to the input image.
    
    Parameters:
        img (numpy.ndarray): Input image.
        
    Returns:
        numpy.ndarray: Image with added Gaussian noise.
    """
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)  # Generate Gaussian noise with mean=0, stddev=25.
    return cv2.add(img, noise)  # Add noise to the original image.

def apply_blur(img, ksize=10):
    """
    Applies Gaussian blur to the image to simulate motion blur.
    
    Parameters:
        img (numpy.ndarray): Input image.
        ksize (int): Size of the kernel (must be a positive odd integer; default is 10).
        
    Returns:
        numpy.ndarray: Blurred image.
    """
    if ksize <= 0 or ksize % 2 == 0:  # Validate kernel size.
        raise ValueError("Kernel size (ksize) must be a positive odd integer.")
    return cv2.GaussianBlur(img, (ksize, ksize), 0)  # Apply Gaussian blur with the specified kernel size.

def rotate_image(img, angle=45):
    """
    Rotates the image by a specified angle around its center.
    
    Parameters:
        img (numpy.ndarray): Input image.
        angle (float): Rotation angle in degrees (default is 45).
        
    Returns:
        numpy.ndarray: Rotated image.
    """
    h, w = img.shape[:2]  # Get the height and width of the image.
    center = (w // 2, h // 2)  # Calculate the center point for rotation.
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # Create the rotation matrix.
    return cv2.warpAffine(img, matrix, (w, h))  # Apply the rotation to the image.

def apply_cutout(img, size=80):
    """
    Applies a random black box (cutout) to the image to simulate occlusion.
    
    Parameters:
        img (numpy.ndarray): Input image.
        size (int): Size of the black box (default is 80).
        
    Returns:
        numpy.ndarray: Image with a cutout applied.
    """
    newimg = img.copy()  # Create a copy of the input image.
    h, w = newimg.shape[:2]  # Get the height and width of the image.
    x1, y1 = np.random.randint(0, w - size), np.random.randint(0, h - size)  # Randomly select the top-left corner.
    newimg[y1:y1 + size, x1:x1 + size] = 0  # Apply the black box.
    return newimg

def distort_image(image, distortion_level):
    """
    Distorts the image by applying a perspective transformation.
    
    Parameters:
        image (numpy.ndarray): Input image.
        distortion_level (int): The degree of distortion.
        
    Returns:
        numpy.ndarray: Distorted image.
    """
    h, w = image.shape[:2]  # Get the height and width of the image.
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])  # Define the source points (corners of the original image).
    dst_pts = np.float32([  # Define the destination points with distortions.
        [0, 0 + distortion_level],
        [w, 0 - distortion_level],
        [w, h + distortion_level],
        [0, h - distortion_level]
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)  # Compute the perspective transformation matrix.
    distorted_image = cv2.warpPerspective(image, M, (w, h))  # Apply the perspective transformation.
    return distorted_image

def process_images_with_augmentations(input_dir, output_dir):
    """
    Processes images from an input directory by applying a series of augmentations 
    and saving the results into subdirectories for each augmentation type.
    
    Parameters:
        input_dir (str): Path to the directory containing the input images.
        output_dir (str): Path to the directory where augmented images will be saved.
        
    Returns:
        None
    """
    # Define augmentation functions and their corresponding names.
    augmentations = {
        "horizontal_flip": lambda img: cv2.flip(img, 1),  # Horizontally flip the image.
        "brightness_contrast": lambda img: adjust_brightness_contrast(img, alpha=3, beta=20),  # Adjust brightness/contrast.
        "gaussian_noise": add_gaussian_noise,  # Add Gaussian noise.
        #"motion_blur": apply_blur,  # (Optional) Apply motion blur.
        "rotation": lambda img: rotate_image(img, angle=45),  # Rotate the image.
        "distortion": lambda img: distort_image(img, distortion_level=60),  # Apply distortion.
        "cutout": apply_cutout  # Apply a random cutout (black box).
    }

    # Create output subdirectories for each augmentation type.
    for aug_name in augmentations.keys():
        aug_output_path = os.path.join(output_dir, aug_name)  # Construct the directory path.
        os.makedirs(aug_output_path, exist_ok=True)  # Create the directory if it doesn't exist.

    # Process each image in the input directory.
    for img_name in tqdm(os.listdir(input_dir), desc="Processing images"):  # Iterate over input images with a progress bar.
        img_path = os.path.join(input_dir, img_name)  # Construct the full file path for the input image.
        image = cv2.imread(img_path)  # Load the image using OpenCV.

        if image is None:  # Skip files that cannot be loaded as images.
            continue

        # Apply each augmentation and save the augmented image.
        for aug_name, aug_function in augmentations.items():
            aug_img = aug_function(image)  # Apply the augmentation function.
            aug_img_name = f"{os.path.splitext(img_name)[0]}_{aug_name}.jpg"  # Construct the augmented image filename.
            aug_img_path = os.path.join(output_dir, aug_name, aug_img_name)  # Construct the augmented image path.
            cv2.imwrite(aug_img_path, aug_img)  # Save the augmented image to the appropriate directory.
