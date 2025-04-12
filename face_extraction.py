import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def extract_faces(image, detection_result, x_extra, y_extra):
    """
    Extracts faces from an image based on detection results and applies padding.

    Parameters:
        image (numpy.ndarray): The input image to be processed.
        detection_result (object): Detection result containing bounding box information.
        x_extra (int): Horizontal padding to be added around detected faces.
        y_extra (int): Vertical padding to be added around detected faces.

    Returns:
        list: A list of face images extracted from the input image.
    """
    cropped_image_list = []  # Initialize an empty list to store cropped face images.
    for detection in detection_result.detections:  # Loop through all detected faces in the image.
        bbox = detection.bounding_box  # Extract bounding box information for the detected face.
        # Compute bounding box coordinates with added padding, ensuring they stay within image boundaries.
        x_start = max(0, bbox.origin_x - x_extra)  # Add padding to the left while avoiding negative coordinates.
        x_end = min(image.shape[1], bbox.origin_x + bbox.width + x_extra)  # Add padding to the right.
        y_start = max(0, bbox.origin_y - y_extra)  # Add padding above the face.
        y_end = min(image.shape[0], bbox.origin_y + bbox.height + y_extra)  # Add padding below the face.
        cropped_image = image[y_start:y_end, x_start:x_end]  # Extract the face region with padding.
        # Check if the cropped face region is valid and large enough for processing.
        if cropped_image.size > 0 and cropped_image.shape[0] > 10 and cropped_image.shape[1] > 10:
            cropped_image_bgr = cv.cvtColor(cropped_image, cv.COLOR_RGB2BGR)  # Convert cropped image to BGR format.
            cropped_image_list.append(cropped_image_bgr)  # Add the processed face image to the list.
    return cropped_image_list  # Return the list of cropped face images.


def process_image(image, x_extra, y_extra, min_confidence):
    """
    Detects faces in an image using BlazeFace via Mediapipe and extracts them with padding.

    Parameters:
        image (numpy.ndarray): The input image to be processed.
        x_extra (int): Horizontal padding to be added around detected faces.
        y_extra (int): Vertical padding to be added around detected faces.
        min_confidence (float): Minimum confidence for considering a detection as a face.

    Returns:
        list: A list of extracted face images from the input image.
    """
    # Configure BlazeFace model options and disable GPU usage (use CPU only).
    base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    options.use_gpu = False
    detector = vision.FaceDetector.create_from_options(options)  # Create a face detector instance.
    detection_result = detector.detect(image)  # Perform face detection on the input image.
    # Filter out detections with confidence scores below the minimum threshold.
    detection_result.detections = [
        det for det in detection_result.detections if det.categories[0].score > min_confidence
    ]
    image_copy = np.copy(image.numpy_view())  # Create a copy of the image for processing.
    face_list = extract_faces(image_copy, detection_result, x_extra, y_extra)  # Extract faces from the image.
    return face_list  # Return the list of extracted face images.


def check_face(image, detector):
    """
    Validates if an input image contains a face using MTCNN detector.

    Parameters:
        image (numpy.ndarray): The input image to be assessed.
        detector (MTCNN): Face detection model (MTCNN).

    Returns:
        bool: True if the image contains a face, False otherwise.
    """
    is_face = True  # Initialize the face validation result as True.
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Convert the image from BGR to RGB for MTCNN compatibility.
    boxes, _ = detector.detect(image_rgb)  # Detect faces using the MTCNN model.
    if boxes is None:  # If no face is detected, update the validation result to False.
        is_face = False
    return is_face  # Return whether the image contains a face.
