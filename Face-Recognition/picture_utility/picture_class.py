class picture_data:
    # picture_class init
    def __init__(self):
        self.album_id = []      # album_id
        self.picture_id = []    # picture_id
        self.status = []        # status (processing)
        self.picture_url = []   # picture_url (http:// ... .jpg)
        self.box = []           # picture_box
        self.encoding = []      # picture_encoding data
        print("생성되었습니다")

    # picture_class destory
    def __del__(self):
        print("삭제되었습니다")

    # file set
    def setAlbumId(self, x, data):
        self.album_id.append(data)

    def setPictureId(self, x, data):
        self.picture_id.append(data)

    def setStatus(self, x, data):
        self.status.append(data)

    def setPictureUrl(self, x, data):
        self.picture_url.append(data)

    def setBox(self, x, data):
        self.box.append(data)

    def setEncoding(self, x, data):
        self.encoding.append(data)

    def ResetBox(self, x, data):
        self.box[x] = data

    def ResetEncoding(self, x, data):
        self.encoding[x] = data  

    # file get
    def getAlbumId(self, x):
        return self.album_id[x]

    def getPictureId(self, x):
        return self.picture_id[x]

    def getStatus(self, x):
        return self.status[x]

    def getPictureUrl(self, x):
        return self.picture_url[x]

    def getLen(self):
        return len(self.album_id)

    def getBox(self, x):
        return self.box[x]

    def getEncoding(self, x):
        return self.encoding[x]