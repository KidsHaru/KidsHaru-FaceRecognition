import json
import requests
import pickle

def ControlFilePickle(data, data2):


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

    picture_url = open(path + "/picture_url.pickle", "wb")
    pickle.dump(data, picture_url)
    picture_url.close()


    box = open(path + "/box.pickle", "wb")
    box.dump(data, box)
    box.close()

    encoding = open(path + "/encoding.pickle", "wb")
    encoding.dump(data, encoding)
    encoding.close()