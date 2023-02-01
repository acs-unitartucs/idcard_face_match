#!/usr/bin/env python3
# Copyright (c) 2021 Burak Can
# Copyright (c) 2022 ACS research group, Institute of Computer Science, University of Tartu
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""This module does comparison of two images"""

from io import BytesIO
from typing import List, Tuple, Union

import math
import numpy as np
from PIL import Image
import cv2


def opencv_dnn_detector() -> cv2.dnn_Net:
    """Create face detection network"""
    if "net" in opencv_dnn_detector.__dict__:
        return opencv_dnn_detector.net

    print("[+] opencv_dnn_detector(): Creating face detector network...")
    # downloaded from (https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel)
    # downloaded from (https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt)
    model_file = "resources/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    config_file = "resources/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)
    # this will try to use CUDA (requires python opencv built with CUDA enabled)
#    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    opencv_dnn_detector.net = net
    return opencv_dnn_detector.net


def get_face_locations(
    image: np.ndarray,
    conf_threshold: float = 0.5,
    scale_size: Tuple[int, int] = (-1, -1),
    non_scaled: bool = False,
) -> Union[List[Tuple[int, ...]], Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]]:
    """Image is expected in opencv format (BGR)
    takes image and returns face location coordinates
    scale_size: Tuple[int, int] (height, width)"""
    # https://learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/
    net = opencv_dnn_detector()


    face_locations: List[Tuple[int, ...]] = []

    if non_scaled:
        face_locations2: List[Tuple[int, ...]] = []

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = detections[0, 0, i, 3]
            y1 = detections[0, 0, i, 4]
            x2 = detections[0, 0, i, 5]
            y2 = detections[0, 0, i, 6]
            if non_scaled:
                x1_ns = int(x1 * image.shape[1])
                y1_ns = int(y1 * image.shape[0])
                x2_ns = int(x2 * image.shape[1])
                y2_ns = int(y2 * image.shape[0])
                face_locations2.append((y1_ns, x2_ns, y2_ns, x1_ns))
            if scale_size == (-1, -1):
                x1 = int(x1 * image.shape[1])
                y1 = int(y1 * image.shape[0])
                x2 = int(x2 * image.shape[1])
                y2 = int(y2 * image.shape[0])
            else:
                x1 = int(x1 * scale_size[1])
                y1 = int(y1 * scale_size[0])
                x2 = int(x2 * scale_size[1])
                y2 = int(y2 * scale_size[0])
            face_locations.append((y1, x2, y2, x1))
    if non_scaled:
        return face_locations, face_locations2
    return face_locations

# 'hog' face detector from dlib
# CPU-based, twice slower, fails at tilted faces
def get_face_locations1(image):
    return face_recognition.face_locations(image, model='hog')

# 'cnn' face detector in dlib
# GPU-based, 100x slower on CPU
def get_face_locations2(image):
    return face_recognition.face_locations(image, model='cnn')

# code from (https://github.com/ageitgey/face_recognition/wiki/Calculating-Accuracy-as-a-Percentage)
# + added check to handle face_distance > 1.0
def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > 1.0:
        return 0
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))
