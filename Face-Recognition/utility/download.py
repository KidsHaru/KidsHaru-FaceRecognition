import os
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

def getLinkDownload(url):
    album_id = Series(np.array([]))
    picture_id = Series(np.array([]))
    picture_name = Series(np.array([]))
    status = Series(np.array([]))
    picture_url = Series(np.array([]))
    box = Series(np.array([]))
    encoding = Series(np.array([]))

    for (path, dir, files) in os.walk(url):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.jpg' or ext == '.png' or ext == '.bmp' or ext == '.jpeg':
                
                # 앨범 ID
                temp_url = path.split("/")
                temp_album = temp_url[len(temp_url) - 1]
                album_id = np.append(album_id, temp_album)

                # 사진 ID
                temp_url = filename.split('.')
                temp_picture = temp_url[0]
                picture_id = np.append(picture_id, temp_picture)
                picture_name = np.append(picture_name, filename)

                # Status
                status = np.append(status, "processing")

                # Picture_URL
                picture_url = np.append(picture_url, path)

                # Picture_BOX
                box = np.append(box, "empty")

                # Picture_Encoding
                encoding = np.append(encoding, "empty")
    
    data = DataFrame({
        'album_id': album_id,
        'picture_id': picture_id,
        'picture_name': picture_name,
        'status': status,
        'picture_url': picture_url,
        'box': box,
        'encoding': encoding
    }, columns=['album_id', 'picture_id', 'picture_name', 'status', 'picture_url', 'box', 'encoding'])

    return data