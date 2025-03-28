import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


#extract_faces: function that takes an image and detction result and isolates the faces in the image. The isolated faces in the image are cut out of the image and appende to a list.
#               This list is returned.
#params
#image:             the image that is assessed
#detection_result:  the detection result object
#x_extra:           the extra pixels that are added in the x-direction
#y_extra:           the extra pixels that are added in the y-direction
#return:            list of face images
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

#process_image: function assess an image and delivers a list with faces from the image.
#               This fuction uses the mediapipe library form google (https://ai.google.dev/edge/mediapipe/solutions/guide). The model that is used is BlazeFace.
#image:             the image that is assessed
#x_extra:           the extra pixels that are added in the x-direction
#y_extra:           the extra pixels that are added in the y-direction
#min_confidence:    a measure for the belief that a subset of the image is a face
#return:            list of face images
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


#check_face: function the performs a check on an image to assess if it is an iamge on a face. The detector is the MTCNN detector from 
#Google FaceNet (https://arxiv.org/abs/1503.03832). Good desciption of the use: https://medium.com/@culuma/face-recognition-with-facenet-and-mtcnn-11e77240adb6
#image:     the image to be assessed
#detector:  the detector to be used
#return:    True if face, False if not a face
def check_face(image,detector):
  is_face = True
  image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
  boxes, _ = detector.detect(image_rgb)
  if boxes is None:
    is_face = False
  return is_face