import urllib.request
import os

def getDownload(album_id, picture_id, file_name, status, picture_url):
    # mkdir
    dirname = "img"
    if not os.path.isdir("./" + dirname + "/"):
    os.mkdir("./" + dirname + "/")

    if not os.path.isdir("./" + album_id + "/"):
    os.mkdir("./" + album_id + "/")

    if not os.path.isdir("./" + picture_id + "/"):
    os.mkdir("./" + picture_id + "/")

    try:
        print(img + "  ", src)
        urllib.request.urlretrieve(img, src + file_name)
        print(img + " 다운로드 완료!")
        return true
    except:
        return false
    