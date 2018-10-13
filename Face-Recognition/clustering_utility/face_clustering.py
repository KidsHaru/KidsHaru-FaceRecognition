from sklearn.cluster import DBSCAN
import pickle
import numpy as np
import os
import signal
import sys
import cv2
import model_custom

def clustering(data2):
    encodings = []
    for i in range(data2.getLen()):
        if data2.getEncoding(i) != []:
            for j in range(len(data2.getEncoding(i))):
                encodings.append(data2.getEncoding(i)[j])

    # print(len(encodings))
    # cluster the embeddings
    clt = DBSCAN(metric="euclidean")
    clt.fit(encodings)

    # label 결정
    label_ids = np.unique(clt.labels_)
    num_unique_faces = len(np.where(label_ids > -1)[0])
    print("clustered %d unique faces." % num_unique_faces)
    # print(len(label_ids))

    for label_id in label_ids:
        dir_name = "ID%d" % label_id

        print(dir_name)

        '''
        os.mkdir(dir_name)

        # find all indexes of label_id
        indexes = np.where(clt.labels_ == label_id)[0]

        # save face images
        for i in indexes:
            frame_id = self.faces[i].frame_id
            box = self.faces[i].box
            pathname = os.path.join(self.capture_dir, self.capture_filename(frame_id))
            
            image = cv2.imread(pathname)
            face_image = self.getFaceImage(image, box)
            filename = dir_name + "-" + self.capture_filename(frame_id)
            pathname = os.path.join(dir_name, filename)
            cv2.imwrite(pathname, face_image)

            print("label_id %d" % label_id, "has %d faces" % len(indexes), "in '%s' directory" % dir_name)
            '''

    print('clustering done')
