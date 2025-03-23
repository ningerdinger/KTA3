import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def extract_faces(image, detection_result, x_extra, y_extra):
  cropped_image_list = []
  for detection in detection_result.detections:
    bbox = detection.bounding_box
    x_start = max(0, bbox.origin_x - x_extra)
    x_end = min(image.shape[1], bbox.origin_x + bbox.width + x_extra)
    y_start = max(0, bbox.origin_y - y_extra)
    y_end = min(image.shape[0], bbox.origin_y + bbox.height + y_extra)
    cropped_image = image[y_start:y_end, x_start:x_end]
    if cropped_image.size > 0 and cropped_image.shape[0] > 10 and cropped_image.shape[1] > 10:           
      cropped_image_bgr = cv.cvtColor(cropped_image, cv.COLOR_RGB2BGR)
      cropped_image_list.append(cropped_image_bgr)
  return cropped_image_list


def process_image(image,x_extra,y_extra, min_confidence):
  base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
  options = vision.FaceDetectorOptions(base_options=base_options)
  options.use_gpu = False
  detector = vision.FaceDetector.create_from_options(options)
  detection_result = detector.detect(image)
  detection_result.detections = [
        det for det in detection_result.detections if det.categories[0].score > min_confidence
    ]
  image_copy = np.copy(image.numpy_view())
  face_list = extract_faces(image_copy,detection_result,x_extra,y_extra)
  return face_list
  