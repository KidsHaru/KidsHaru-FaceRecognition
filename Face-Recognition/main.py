import json
import requests
import pickle
from picture_utility import picture_class as pc
from picture_utility import picture_download as pdl
from picture_utility import picture_detect as pdt
from picture_utility import picture_pickle as pp

# https://docs.google.com/document/d/1lwofKqyqlq--8LuiqjpFgP4R-7mPNt0JT5QwAcu_MRA/edit
# FaceRecognition <-> Server

# https://kidsharu.github.io/KidsHaru-APIDoc/
# 자세한 API 문서

# picture path 확인, 나누기
data = pc.picture_data()
cnt = pdl.getDownload(data)

# picture detecting, encoding
for i in range(cnt):
    album_id = data.getAlbumId(i).strip()
    picture_id = data.getPictureId(i).strip()
    status = data.getStatus(i).strip()
    picture_url = data.getPictureUrl(i).strip()

    box, encoding = pdt.faceDetect(album_id, picture_id, status, picture_url)
    # print(box, encoding)

    if box == -1 and encoding == -1:
        print('이미 처리된 파일입니다.')
    else:
        data.ResetBox(i, box)
        data.ResetEncoding(i, encoding)

data2 = pc.picture_data()
for i in range(cnt):
    if(data.getBox(i) != '-' and data.getEncoding(i) != '-'):
        print('테스트 성공!')
    else:
        pp.ControlFilePickle(data, data2)
'''
# picture class pickle 저장
path = "./picture_utility/picture_pickle"
if not os.path.isdir(path):    # 디렉토리 유무 확인
    os.mkdir(path)             # 없으면 생성하라

album_id = open(path + "/album_id.pickle", "wb")
pickle.dump(data, album_id)
album_id.close()

picture_id = open(path + "/picture_id.pickle", "wb")
pickle.dump(data, picture_id)
picture_id.close()

status = open(path + "/status.pickle", "wb")
pickle.dump(data, status)
status.close()

box = open(path + "/box.pickle", "wb")
box.dump(data, box)
box.close()

encoding = open(path + "/encoding.pickle", "wb")
encoding.dump(data, encoding)
encoding.close()
'''


'''
file = open("./picture_utility/picture_pickle.txt", "rb")
print(pickle.load(file))
# print(file)
'''
