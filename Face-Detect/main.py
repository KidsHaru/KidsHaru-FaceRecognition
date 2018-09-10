import requests
import picture_class
import picture_download

# url get
try:
    url = 'https://fc3i3hiwel.execute-api.ap-northeast-2.amazonaws.com/develop/pictures/processing'
    response = requests.get(url)
except:
    print("URL 주소가 잘못되었습니다")

# data get
try:
    data = picture_class.picture_data()
    data.append_data(response)
except:
    print("사이트를 읽는데 실패하였습니다")

# picture download
try:
    for i in range(1):
        album_id = str(data.getAlbumId(i)).strip()
        picture_id = str(data.getAlbumId(i)).strip()
        file_name = str(data.getFileName(i)).strip()
        status = data.getStatus(i)
        picture_url = str(data.getPictureUrl(i)).strip()
        
        picture_download.getDownload(album_id, picture_id, file_name, status, picture_url)
except:
    print("사진을 다운로드하는데 실패하였습니다.")
# ata.print_id(0, 1)