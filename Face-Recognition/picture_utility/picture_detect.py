from sklearn.cluster import DBSCAN
import pickle
import numpy as np
import os
import signal
import sys
import cv2
import model_custom

def faceDetect(album_id, picture_id, status, picture_url):
    # (top, right, bottom, left)
    global top, right, bottom, left

    # 이미지 인식
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread(picture_url)

    # HOG 모델로 face_detecting 완료
    box = model_custom.face_locations(img, model="hog")

    # Rectangle 사각형
    for top, right, bottom, left in box:
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

    # Encoding
    encoding = model_custom.face_encodings(img, box)

    # write
    # dirname = "picture_after"
    # re_url = "./picture_utility/picture_after\\" + album_id + "/"

    # if not os.path.isdir(re_url):
        # os.mkdir(re_url)
    # print(re_url)

    # message
    print('Face Detecting 완료!')

    return box, encoding