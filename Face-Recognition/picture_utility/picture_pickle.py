import json
import requests
import pickle
import gzip

def ReadPickle(data2):
    path = "./picture_utility/picture_pickle/picture_class.pickle"
    file = gzip.open(path, "rb")
    temp = pickle.load(file)
    # print(temp)
    file.close()
    
    for i in range(temp.getLen()):
        data2.setAlbumId(i, temp.getAlbumId(i))
        data2.setPictureId(i, temp.getPictureId(i))
        data2.setStatus(i, temp.getStatus(i))
        data2.setPictureUrl(i, temp.getPictureUrl(i))
        data2.setBox(i, temp.getBox(i))
        data2.setEncoding(i, temp.getEncoding(i))
        # print('데이터 불러오기 완료!')


def WritePickle(data2):
    path = "./picture_utility/picture_pickle/picture_class.pickle"
    file = gzip.open(path, "wb")
    pickle.dump(data2, file)
    # print(file)
    file.close()


def WriteAppendFile(data, data2, i):
    data2.setAlbumId(data2.getLen(), data.getAlbumId(i))
    data2.setPictureId(data2.getLen(), data.getPictureId(i))
    data2.setStatus(data2.getLen(), data.getStatus(i))
    data2.setPictureUrl(data2.getLen(), data.getPictureUrl(i))
    data2.setBox(data2.getLen(), data.getBox(i))
    data2.setEncoding(data2.getLen(), data.getEncoding(i))
    # print('데이터 저장 완료!')