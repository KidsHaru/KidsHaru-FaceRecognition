from sklearn.cluster import DBSCAN
import pickle
import numpy as np
import os
import signal
import sys
import cv2
import model_custom

'''
img = cv2.imread("./image_file/test/davichi.jpg")
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


file = open("./picture_utility/picture_pickle.txt", "wb")
pickle.dump("Test", file)
file.close()

file = open("./picture_utility/picture_pickle.txt", "rb")
print(pickle.load(file))
'''