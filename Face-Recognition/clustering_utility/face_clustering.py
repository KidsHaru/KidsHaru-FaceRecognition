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
        encodings.append(data2.getEncoding(i))

    # cluster the embeddings
    clt = DBSCAN(metric="euclidean")
    clt.fit(encodings)

    # label 결정
    label_ids = np.unique(clt.labels_)



'''
def cluster(self):
        if len(self.faces) is 0:
            print("no faces to cluster")
            return

        print("start clustering %d faces..." % len(self.faces))
        encodings = [face.encoding for face in self.faces]

        # cluster the embeddings
        clt = DBSCAN(metric="euclidean")
        clt.fit(encodings)

        # determine the total number of unique faces found in the dataset
        label_ids = np.unique(clt.labels_)
        num_unique_faces = len(np.where(label_ids > -1)[0])
        print("clustered %d unique faces." % num_unique_faces)

        os.system("rm -rf ID*")
        for label_id in label_ids:
            dir_name = "ID%d" % label_id
            os.mkdir(dir_name)

            # find all indexes of label_id
            indexes = np.where(clt.labels_ == label_id)[0]

            # save face images
            for i in indexes:
                frame_id = self.faces[i].frame_id
                box = self.faces[i].box
                pathname = os.path.join(self.capture_dir,
                                        self.capture_filename(frame_id))
                image = cv2.imread(pathname)
                face_image = self.getFaceImage(image, box)
                filename = dir_name + "-" + self.capture_filename(frame_id)
                pathname = os.path.join(dir_name, filename)
                cv2.imwrite(pathname, face_image)

            print("label_id %d" % label_id, "has %d faces" % len(indexes),
                  "in '%s' directory" % dir_name)

        print('clustering done')
'''