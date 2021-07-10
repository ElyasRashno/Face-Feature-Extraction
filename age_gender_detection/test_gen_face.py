import re
import time
from datetime import datetime
from scipy.io import loadmat
import cv2
import json
import mxnet as mx
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
from detect.mx_mtcnn.mtcnn_detector import MtcnnDetector


from PIL import Image
import matplotlib.pyplot as plt
MTCNN_DETECT = MtcnnDetector(model_folder=None, ctx=mx.cpu(0), num_worker=1, minsize=50, accurate_landmark=True)

def gen_face(detector, image, image_path="", only_one=True):


    imgplot = plt.imshow(image)

    ret = detector.detect_face(image)
    if not ret:
        raise Exception("cant detect facei: %s"%image_path)
    bounds, lmarks = ret
    if only_one and len(bounds) > 1:
        print("!!!!!,", bounds, lmarks)
        raise Exception("more than one face %s"%image_path)
    return ret

#img = cv2.imread("/home/user/Desktop/face-biometric-server/data/face_male/Google_0424.jpeg")
img = cv2.imread("ssss.jpg")

bounds, lmarks = gen_face(MTCNN_DETECT, img, only_one=False)
print(bounds, "\n", lmarks)