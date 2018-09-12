import numpy as np
import cv2
import sys
import urllib.request
import os

def faceDetect(album_id, picture_id, file_name, status, picture_url):
    global x, y, w, h

    checking = "img_before"
    dirname = "img_after"                         # 체크하고자 하는 디렉토리명
    if not os.path.isdir("./" + dirname + "/"):    # 디렉토리 유무 확인
        os.mkdir("./" + dirname + "/")             # 없으면 생성하라

    dirname += "/" + album_id
    checking += "/" + album_id
    if not os.path.isdir("./" + dirname + "/"):    # 디렉토리 유무 확인
        os.mkdir("./" + dirname + "/")             # 없으면 생성하라

    dirname += "/" + picture_id
    checking += "/" + picture_id
    if not os.path.isdir("./" + dirname + "/"):    # 디렉토리 유무 확인
        os.mkdir("./" + dirname + "/")             # 없으면 생성하라

    dirname += "/" + file_name
    checking += "/" + file_name
    print(dirname)

    if os.path.isfile("./" + dirname):
        print("파일이 이미 있습니다")
        return -1

    ###################################################
    font = cv2.FONT_HERSHEY_SIMPLEX
    cascPath = "haarcascade_frontface.xml"

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)
    
    # 계산 반복 횟수 (한번만 처리하려면 아래를 1로 하거나 for문을 제거하세요)
    iteration_count = 1
    for cnt in range(0, iteration_count):
        # Read the image
        image = cv2.imread("./" + checking)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,     # 이미지에서 얼굴 크기가 서로 다른 것을 보상해주는 값
            minNeighbors=3,    # 얼굴 사이의 최소 간격(픽셀)입니다
            minSize=(10, 10),   # 얼굴의 최소 크기입니다
        )

        # 검출된 얼굴 주변에 사각형 그리기
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            print(x, y, w, h)
        
        # 얼굴을 검출한 이미지를 화면에 띄웁니다
        # cv2.imshow("Face Detected", image)
    
    # cv2.imwrite(picture_url, dirname)
    print('처리 성공!')
    return 1
    
    