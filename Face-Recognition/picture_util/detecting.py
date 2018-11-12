import cv2
import numpy as np
import model_custom

def faceDetect(data):
    global top, right, bottom, left

    # 이미지 인식
    dirname = data['picture_url'] + "/" + data['picture_name']
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread(dirname)

    # 높이 700으로 image 사이즈 조정
    height, width, channel = img.shape
    x = 700 / height
    img = cv2.resize(img, dsize=(0, 0), fx= x, fy= x, interpolation=cv2.INTER_LINEAR)

    # 노말라이징
    dst = np.zeros(shape=(5,2))
    norm_img = cv2.normalize(img, dst, 0, 255, cv2.NORM_MINMAX)

    # CNN 모델로 face_detecting 완료
    box = model_custom.face_locations(norm_img, model="hog")
    print(box)

    # Encoding
    encoding = model_custom.face_encodings(norm_img, box)
    # print(encoding)

    # cv2.imshow("test", norm_img)
    # cv2.waitKey(0)
    
    return box, encoding

    