import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pandas import Series, DataFrame
from utility import path, download, pickle
from picture_util import detecting, clustering, saving

# =========================================
# 이미지 다운로드
url = path.getDirname("image")
data = download.getLinkDownload(url)

# =========================================
# 이미지 저장
url = path.getDirname("pickle_data") + "/picture_pickle.pickle"
data_temp, data = saving.picture_saving(url, data)

# print(len(data), len(data_temp))

# =========================================
# 재 로드 하기
data = pickle.ReadPickle(url)
print('pickle 로드 완료!')

index_url = path.getDirname("pickle_data") + "/index_pickle.pickle"
box_url = path.getDirname("pickle_data") + "/box_pickle.pickle"
encoding_url = path.getDirname("pickle_data") + "/encoding_pickle.pickle"

indexE_url = path.getDirname("pickle_data") + "/indexE_pickle.pickle"
encodings_url = path.getDirname("pickle_data") + "/encodings_pickle.pickle"

index = saving.util_saving("index", index_url)
box = saving.util_saving("box", box_url)
encoding = saving.util_saving("encoding", encoding_url)

indexE = saving.util_saving("indexE", indexE_url)
encodings = saving.util_saving("encodings", encodings_url)

# =========================================
# 이미지 detecting

# data.loc[data.index == 1, 'encoding'] = 'complete'
data_temp = data.loc[data['encoding'] == "empty"]
result = data_temp.index

for x in range(0, 10):
    if x in result:
        box_t, encoding_t = detecting.faceDetect(data.ix[x])   

        if len(box_t) > 0 and len(encoding_t) > 0:
            index.append(x)
            box.append(box_t)
            encoding.append(encoding_t)

            temp_len = len(index) - 1
            for y in range(len(box[temp_len])):
                indexE.append(str(x) + "." + str(y))
                encodings.append( encoding[temp_len][y] )

            data.loc[data.index == x, 'box'] = len(box[temp_len])
            data.loc[data.index == x, 'encoding'] = "complete"

            clustering.cluster(encodings)

print(index)
print(indexE)
print(data)


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