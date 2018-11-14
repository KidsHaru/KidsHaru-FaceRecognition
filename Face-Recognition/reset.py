import json
import requests

url = "https://gimic6gh9i.execute-api.ap-northeast-2.amazonaws.com/develop/faces/reset"
json_data = {}

json_string = json.dumps(json_data).encode("utf-8")
response = requests.post(url, data=json_string)
print(response)