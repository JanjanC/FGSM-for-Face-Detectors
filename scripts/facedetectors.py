import scripts.utils as utils
import typing
import os
import cv2
from pytorchyolo import detect, models
import mediapipe as mp
import numpy as np

WEIGHTS_DIR = os.path.join(os.getcwd(), 'face_detector_weights')

class BoxCoords(typing.NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int

class MediaPipe():
    def __init__(self, conf=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.conf = conf
    
    def detect(self, image, return_boxes = True):
        image = utils.tensor_to_np(image)
        
        with self.mp_face_detection.FaceDetection(min_detection_confidence=self.conf, model_selection=0) as face_detection:
            results = face_detection.process(image)
            if not return_boxes:
                return results.detections is not None
            else:
                if results.detections is None:
                    return []
                    
                bboxes = []
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    bbox = [bbox.xmin, bbox.ymin, bbox.xmin + bbox.width, bbox.ymin + bbox.height]
                    bbox = list(map(lambda n: np.clip(n, 0., 1.), bbox))
                    height, width, _ = image.shape
                    bbox[0], bbox[1] = self.mp_drawing._normalized_to_pixel_coordinates(bbox[0], bbox[1], width, height)
                    bbox[2], bbox[3] = self.mp_drawing._normalized_to_pixel_coordinates(bbox[2], bbox[3], width, height)
                    bboxes.append(BoxCoords(*bbox))
                    
                return bboxes

class YuNet():
    def __init__(self, nms=None, conf=None, weight_loc=os.path.join(WEIGHTS_DIR, "YuNet", "yunet.onnx"), weight_url="https://drive.google.com/uc?id=1V7FdMzjwyGPn5QwW18D4cirzbZfKwC-i"):
        self.nms = nms
        self.conf = conf
        utils.download_weight(weight_loc, weight_url)
        self.yn_face_detector = cv2.FaceDetectorYN_create(weight_loc, "", (0, 0))

    def detect(self, image, return_boxes = True):
        image = utils.tensor_to_np(image)
        height, width, _ = image.shape
        self.yn_face_detector.setInputSize((width, height))
        if self.nms is not None:
            self.yn_face_detector.setNMSThreshold(self.nms)
        if self.conf is not None:
            self.yn_face_detector.setScoreThreshold(self.conf)
        _, faces = self.yn_face_detector.detect(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not return_boxes:
            return faces is not None
        else:
            if faces is None:
                return []
            bboxes = [
                BoxCoords(
                    int(np.floor(face[0])),
                    int(np.floor(face[1])),
                    int(np.ceil(face[0] + face[2])),
                    int(np.ceil(face[1] + face[3]))
                ) for face in faces
            ]
            return bboxes

class YoloFace():
    def __init__(self, nms=0.5, conf=0.5,
        cfg_loc=os.path.join(WEIGHTS_DIR, "YOLOFace", "yoloface.cfg"), cfg_url="https://drive.google.com/uc?id=1CPUZlYL5ik4d9y6oCyzi0930KgzawI6V",
        weight_loc=os.path.join(WEIGHTS_DIR, "YOLOFace", "yoloface.weights"), weight_url="https://drive.google.com/uc?id=1utquM5TAnfIa1Aq0X9fCvrllHiTWazdD"):
        self.nms = nms
        self.conf = conf
        utils.download_weight(weight_loc, weight_url)
        utils.download_weight(cfg_loc, cfg_url)
        _, self.yf_face_detector = models.load_model(cfg_loc, weight_loc)

    def detect(self, image, return_boxes = True):
        image = utils.tensor_to_np(image)
        bboxes = detect.detect_image(self.yf_face_detector, image, conf_thres=0.5, nms_thres=0.5)
        if not return_boxes:
            return bboxes is not None
        else:
            return [BoxCoords(*map(int, bbox[:4])) for bbox in bboxes]