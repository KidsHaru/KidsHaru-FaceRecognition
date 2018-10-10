import model_custom
import cv2
from sklearn.cluster import DBSCAN
import numpy as np
import os
import pickle
import signal
import sys

img = cv2.imread("davichi.jpg")
boxed = model_custom.face_locations(img, model="hog")
# print(boxed)

encodings = model_custom.face_encodings(img, boxed)
# print(encodings)

################################################################
clt = DBSCAN(metric="euclidean")
clt.fit(encodings)

label_ids = np.unique(clt.labels_)
num_unique_faces = len(np.where(label_ids > -1)[0])
print(num_unique_faces)