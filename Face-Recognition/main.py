import os
import json
import requests
from picture_utility import picture_class as pc
from picture_utility import picture_download as pd
from picture_utility import picture_search as ps 
from picture_utility import picture_detect as pd

# https://docs.google.com/document/d/1lwofKqyqlq--8LuiqjpFgP4R-7mPNt0JT5QwAcu_MRA/edit
# FaceRecognition <-> Server

# https://kidsharu.github.io/KidsHaru-APIDoc/
# 자세한 API 문서

path_dir = "./picture_utility/picture_before"
for (path, dir, files) in os.walk(path_dir):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.jpg' or ext == '.png':
            print("%s/%s" % (path, filename))

'''
# picture download - 수정 필요
try:
    for i in range(0, data.getLen()):
        album_id = str(data.getAlbumId(i)).strip()
        picture_id = str(data.getPictureId(i)).strip()
        status = data.getStatus(i)
        picture_url = str(data.getPictureUrl(i)).strip()
        # print(album_id, picture_id, status, picture_url)
        download = pd.getDownload(album_id, picture_id, status, picture_url)
except:
    print("사진을 저장하는데 실패하였습니다.")
'''