import requests
import json

url = 'http://192.168.31.197:9777/generate_anno'
post_data = {
    'algo_type': "smoking_calling",
    'video_paths': [],
    'image_paths': ["data/person/005.jpg", "data/person/009.jpg"],
    'out_root': "data/main_result/gesture",
}
response = requests.post(url, json=post_data)
# response = requests.get(url)
print(json.loads(response.content))
