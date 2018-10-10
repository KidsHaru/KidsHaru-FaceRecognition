# -*- coding: utf-8 -*-

import PIL.Image
import dlib
import numpy as np

# face_recognition 모델 사용
try:
    import face_recognition_models
except Exception:
    print("pip install git+https://github.com/ageitgey/face_recognition_models")
    quit()

face_detector = dlib.get_frontal_face_detector()

face_detector = dlib.get_frontal_face_detector()

predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

predictor_5_point_model = face_recognition_models.pose_predictor_five_point_model_location()
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

# dlib 'rect' 객체를 top, right, bottom, left 순서로 변환
# param rect : dlib 'rect' 오브젝트
# return : tuple (top, right, bottom, left)
def _rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()

# tuple (top, right, bottom, left) -> dlib 'rect' 객체 변환
# param css : tuple (t, r, b, l)
# return : dlib 'rect' 오브젝트
def _css_to_rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])

# (top, right, bottom, left) 순서로 dlib 'rect' 오브젝트가 지정된 범위에 있는지 확인
# param css : tuple (t, r, b, l)
# param image_shape : numpy image array
# return : tuple (t, r, b, l)
def _trim_css_to_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

# 얼굴 인코딩의 목록이 주어지면, 유클리드상 거리를 구한다.
# 거리는 얼굴들 간에 얼마나 비슷한지 알려준다.
# param face_encodings : 비교할 얼굴 인코딩 목록
# param face_to_compare : 비교할 얼굴 인코딩
# return : 각 면의 길이를 반환 (numpy array)
def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)

# .jpg, .png 파일을 numpy array으로 업로드 한다.
# param file : 로드할 이미지 파일 이름 또는 객체
# param mode : 이미지를 변환할 형식 (RGB혹은 L 가능)
# return : 이미지파일을 numpy array로 변환
def load_image_file(file, mode='RGB')
    im = PIL.Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)

# image에서 경계 상자 배열을 반환한다.
# param img : 이미지 (numpy array)
# param number_of_time_to_upsample : 면을 찾는 이미지를 sampling 하는 횟수, 많아질 수록 사진이 작아진다
# model : 사용할 얼굴 검출 모델, cnn이 더 정확하나, 여기서는 HOG 알고리즘을 사용한다.
# return : 검출한 경계 상자 배열 -> dlib 상자 리스트
def _raw_face_locations(img, number_of_times_to_upsample=1, model="hog"):
    if model == "cnn":
        return cnn_face_detector(img, number_of_times_to_upsample)
    else:
        return face_detector(img, number_of_times_to_upsample)

# image에서 경계 상자 배열을 반환한다.
# param img : 이미지 (numpy array)
# param number_of_time_to_upsample : 면을 찾는 이미지를 sampling 하는 횟수, 많아질 수록 사진이 작아진다
# model : 사용할 얼굴 검출 모델, cnn이 더 정확하나, 여기서는 HOG 알고리즘을 사용한다.
# return : 검출한 경계 상자 배열 -> dlib 상자 리스트
def face_locations(img, number_of_times_to_upsample=1, model="hog"):
    if model == "cnn":
        return [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample, "cnn")]
    else:
        return [_trim_css_to_bounds(_rect_to_css(face), img.shape) for face in _raw_face_locations(img, number_of_times_to_upsample, model)]

# cnn 얼굴 검출을 사용해서 영상의 사람 2d 배열이 반환된다
# param img : 이미지 (numpy array)
# param number_of_time_to_upsample : 면을 찾는 이미지를 sampling 하는 횟수, 많아질 수록 사진이 작아진다
# return : 검출한 경계 상자 배열 -> dlib 상자 리스트
def _raw_face_locations_batched(images, number_of_times_to_upsample=1, batch_size=128):
    return cnn_face_detector(images, number_of_times_to_upsample, batch_size=batch_size)


def batch_face_locations(images, number_of_times_to_upsample=1, batch_size=128):
    def convert_cnn_detections_to_css(detections):
        return [_trim_css_to_bounds(_rect_to_css(face.rect), images[0].shape) for face in detections]

    raw_detections_batched = _raw_face_locations_batched(images, number_of_times_to_upsample, batch_size)

    return list(map(convert_cnn_detections_to_css, raw_detections_batched))