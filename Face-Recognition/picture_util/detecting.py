import cv2
import numpy as np
import model_custom

def faceDetect(data):
    global top, right, bottom, left

    # 이미지 인식
    dirname = data['picture_url'] + "/" + data['picture_name']
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread(dirname)
    
    # CNN 모델로 face_detecting 완료
    box = model_custom.face_locations(img, model="cnn")
    print(box)

    # Encoding
    encoding = model_custom.face_encodings(img, box)
    print(encoding)
    
    return box, encoding

    