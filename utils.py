import numpy as np
import cv2 as cv
import mediapipe as mp
import os


def extract_frames(video_file_path, video_name, sample_every):
    """
    Extracts frames from a video file at specified intervals.

    Parameters:
        video_file_path (str): Path to the directory containing the video file.
        video_name (str): Name of the video file to be processed.
        sample_every (int): Number of frames to skip between samples.

    Returns:
        list: A list of frames extracted from the video in Mediapipe's image format.
    """
    print('Start extracting frames')  # Indicate the start of frame extraction.
    frame_list = []  # Initialize an empty list to store extracted frames.
    video_path = os.path.join(video_file_path, video_name)  # Construct the full path to the video file.
    cap = cv.VideoCapture(video_path)  # Open the video file for reading.
    
    if not cap.isOpened():  # Check if the video file was successfully opened.
        print("Error: Cannot open video file.")  # Print an error message if the video cannot be opened.
        return []  # Return an empty list if the video fails to open.

    i = 0  # Initialize a frame counter.
    captured = 0  # Counter for successfully captured frames.
    
    while cap.isOpened():  # Continue reading frames while the video is open.
        ret, frame = cap.read()  # Read a frame from the video.
        if not ret:  # Break the loop if there are no more frames to read.
            break
        if i % sample_every == 0:  # Check if the current frame matches the sampling interval.
            frame_rgb = np.ascontiguousarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))  # Convert the frame to RGB format.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)  # Convert to Mediapipe's image format.
            frame_list.append(mp_image)  # Append the frame to the list.
            captured += 1  # Increment the count of captured frames.
        i += 1  # Increment the frame counter.
    
    cap.release()  # Release the video capture object.
    cv.destroyAllWindows()  # Close any OpenCV windows.
    print(f"Successfully captured {captured} frames")  # Print the number of captured frames.
    return frame_list  # Return the list of extracted frames.


def save_face_list(face_list_movie, faces_folder, movie, extension):
    """
    Saves a list of face images to a specified folder.

    Parameters:
        face_list_movie (list): List of face images to be saved.
        faces_folder (str): Path to the folder where face images will be saved.
        movie (str): Name of the movie (used for naming files).
        extension (str): File extension for saved images (e.g., '.png' or '.jpg').

    Returns:
        None
    """
    saved_faces = 0  # Counter for saved face images.
    for i, face in enumerate(face_list_movie):  # Iterate through the list of face images.
        if face.shape[0] > 10 and face.shape[1] > 10:  # Check if the face image is large enough to save.
            cv.imwrite(faces_folder + movie + '_' + str(saved_faces) + extension, face)  # Save the face image to the folder.
            saved_faces += 1  # Increment the count of saved faces.
    return  
