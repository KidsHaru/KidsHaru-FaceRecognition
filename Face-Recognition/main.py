import json
import requests
from picture_utility import picture_class as pc
from picture_utility import picture_download as pd

# https://docs.google.com/document/d/1lwofKqyqlq--8LuiqjpFgP4R-7mPNt0JT5QwAcu_MRA/edit
# FaceRecognition <-> Server

# https://kidsharu.github.io/KidsHaru-APIDoc/
# 자세한 API 문서

url = "https://fc3i3hiwel.execute-api.ap-northeast-2.amazonaws.com/develop/pictures/processing"

'''
# data get - 수정 필요
try:
    # URL에서 정보 얻기
    response = requests.get(url)

    # 데이터 추가
    data = pc.picture_data()
    data.append_data(response)
except:
    print("URL 주소 인식 실패")
'''

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


