import json
import requests
from picture_utility import picture_class as pc
from picture_utility import picture_download as pdl
from picture_utility import picture_detect as pdt

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
    data.ResetBox(i, box)
    data.ResetEncoding(i, encoding)