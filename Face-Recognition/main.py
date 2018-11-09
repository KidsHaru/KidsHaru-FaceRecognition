import os
import sys
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from utility import path, download, pickle
from picture_util import detecting

# =========================================
# 이미지 다운로드
url = path.getDirname("image")
data = download.getLinkDownload(url)

# =========================================
# 이미지 저장
temp = DataFrame({})
url = path.getDirname("pickle_data") + "/picture_pickle.pickle"
print(os.path.exists(url))
if os.path.isfile(url):
    data_temp = pickle.ReadPickle(url)
    print('pickle 로드 완료!')
else:
    pickle.WritePickle(url, temp)
    data_temp = 1
    print('pickle 저장 완료!')

# =========================================
box = []
encoding = []

# 이미지 detecting
for x in range(2, 3):
    box_t, encoding_t = detecting.faceDetect(data.ix[x])
    box.append(box_t)
    encoding.append(encoding_t)

box_nd = np.append(box, [])
print(box)









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