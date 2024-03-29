import requests
import json
import threading


def temp_worker(url, data):
    response = requests.post(url, json=data)
    print(json.loads(response.content))


url_anno = 'http://localhost:19777/generate_anno'

url_vio = 'http://localhost:19777/decode_video'

post_data_anno = {
    'algo_type': "detcls",
    'video_paths': [],
    'image_paths': ["data/person/005.jpg", "data/person/004.jpg"],
    'out_root': "data/output0",
}

post_data_vio = {
    'out_root': "data/output1",
    'video_paths': ["rtsp://admin:xsqjyz888@114.242.23.39:6024/Streaming/Channels/101", "rtsp://admin:xsqjyz888@114.242.23.39:6021/Streaming/Channels/101", "rtsp://admin:xsqjyz888@114.242.23.39:6026/Streaming/Channels/101"],
    'interval': 5,
    'times': 10,
}

# 模拟多客户端调用
print('thread %s is running...' % threading.current_thread().name)
t0 = threading.Thread(target=temp_worker, args=(
    url_anno, post_data_anno,), name='LoopThread0')
t1 = threading.Thread(target=temp_worker, args=(
    url_vio, post_data_vio,), name='LoopThread1')

t0.start()
t1.start()

t0.join()
t1.join()
