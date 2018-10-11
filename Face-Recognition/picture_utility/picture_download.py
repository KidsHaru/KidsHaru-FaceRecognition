import urllib.request
import os

def getDownload(data):
    cnt = 0
    path_dir = "./picture_utility/picture_before"
    for (path, dir, files) in os.walk(path_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.jpg' or ext == '.png':
                album_id = path.split("\\")[1]
                # print(path)
                # print(album_id)
                # print(filename)

                cnt += 1
                data.setAlbumId(cnt, album_id)
                data.setPictureId(cnt, filename)
                data.setStatus(cnt, "processing")
                data.setPictureUrl(cnt, path + "/" + filename)
                data.setBox(cnt, "-")
                data.setEncoding(cnt, "-")

    return cnt