import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pandas import Series, DataFrame
from utility import path, download, pickle
from picture_util import detecting, clustering


# =========================================
# 이미지 다운로드
url = path.getDirname("image")
data = download.getLinkDownload(url)

# =========================================
# 이미지 저장
data_temp = DataFrame({})
temp = DataFrame({})
url = path.getDirname("pickle_data") + "/picture_pickle.pickle"
if os.path.isfile(url):
    data_temp = pickle.ReadPickle(url)
    print('pickle 로드 완료!')

    if len(data_temp) > 0:
        for x in range(len(data)):
            url_temp = data.ix[x]['picture_url']
            name_temp = data.ix[x]['picture_name']

            arr2 = []
            try:
                arr1 = data_temp.loc[data_temp['picture_url'] == url_temp]
                if len(arr1) > 0:
                    arr2 = arr1.loc[arr1['picture_name'] == name_temp]
            except:
                pass
            
            if len(arr2) < 1:
                data_temp = data_temp.append(data.ix[x], ignore_index = True)
        
        pickle.WritePickle(url, data_temp)
        print('pickle 저장 완료!')

    else:
        pickle.WritePickle(url, data)
        print('pickle 저장 완료!')

else:
    pickle.WritePickle(url, temp)
    print('pickle 저장 완료!')

# print(len(data), len(data_temp))

# =========================================
# index, box, encoding 저장
index = []
box = []
encoding = []
encodings = []

index_url = path.getDirname("pickle_data") + "/index_pickle.pickle"
box_url = path.getDirname("pickle_data") + "/box_pickle.pickle"
encoding_url = path.getDirname("pickle_data") + "/encoding_pickle.pickle"
encodings_url = path.getDirname("pickle_data") + "/encodings_pickle.pickle"

if os.path.isfile(index_url):
    index = pickle.ReadPickle(index_url)
    print('pickle 로드 완료!')

    if len(index) > 0:
        for x in range(len(data)):
            url_temp = data.ix[x]['picture_url']
            name_temp = data.ix[x]['picture_name']

            arr2 = []
            try:
                arr1 = data_temp.loc[data_temp['picture_url'] == url_temp]
                if len(arr1) > 0:
                    arr2 = arr1.loc[arr1['picture_name'] == name_temp]
            except:
                pass
            
            if len(arr2) < 1:
                data_temp = data_temp.append(data.ix[x], ignore_index = True)
        
        pickle.WritePickle(url, data_temp)
        print('pickle 저장 완료!')

    else:
        pickle.WritePickle(url, data)
        print('pickle 저장 완료!')

else:
    pickle.WritePickle(url, temp)
    print('pickle 저장 완료!')


# =========================================
# 이미지 detecting

# 재 로드 하기
data = pickle.ReadPickle(url)
print('pickle 로드 완료!')

data.loc[data.index == 1, 'encoding'] = 'complete'
data_temp = data.loc[data['encoding'] == "empty"]
result = data_temp.index


for x in range(0, 10):
    # data 완료 구분
    url_temp = data.ix[x]['picture_url']
    name_temp = data.ix[x]['picture_name']

    if x in result:
        box_t, encoding_t = detecting.faceDetect(data.ix[x])

        if len(box_t) > 0 and len(encoding_t) > 0:
            index.append(x)
            box.append(box_t)
            encoding.append(encoding_t)

            for y in range(len(box[len(index) - 1])):
                # print(box[len(index) - 1][y])
                encodings.append( encoding[len(index) - 1][y] )

            data.loc[data.index == x, 'box'] = len(box[len(index) - 1])
            data.loc[data.index == x, 'encoding'] = "complete"

            clustering.cluster(encodings)
    
    else:
        pass


# print(data_temp)
# print(encodings)
# print(data)







# result = data.index
# print(result[70])

# data.loc[data.index == 710, 'box'] = 'test'
# print(data.ix[1]['box'])

'''
if __name__ == "__main__":
    if len(sys.argv) - 1:
        if(sys.argv[1] == '-d' or sys.argv[1] == '--download'):
            print('다운로드 시작')
        else:
            print('명령 인자를 바르게 입력해주세요')

    else:
        print('명령 인자를 바르게 입력해주세요')
'''