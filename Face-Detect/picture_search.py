import urllib.request
import os

def faceSearch(album_id, picture_id, file_name, status, picture_url):
    img_before = "img_before"
    img_after = "img_after"                         # 체크하고자 하는 디렉토리명
    if not os.path.isdir("./" + img_after + "/"):    # 디렉토리 유무 확인
        os.mkdir("./" + img_after + "/")             # 없으면 생성하라

    img_before += "/" + album_id
    img_after += "/" + album_id
    if not os.path.isdir("./" + img_after + "/"):    # 디렉토리 유무 확인
        os.mkdir("./" + img_after + "/")             # 없으면 생성하라

    img_before += "/" + picture_id
    img_after += "/" + picture_id
    if not os.path.isdir("./" + img_after + "/"):    # 디렉토리 유무 확인
        os.mkdir("./" + img_after + "/")             # 없으면 생성하라

    img_before += "/" + file_name
    img_after += "/" + file_name
    # print(img_after)

    if os.path.isfile("./" + img_after):
        print("파일이 이미 있습니다")
        return -1, -1
    
    return img_before, img_after