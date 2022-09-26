from flask import Flask, jsonify, request
import itertools
import uuid
import tqdm
import cv2
import os
from gevent import pywsgi
from params import smoking_calling_gesture_params
from module import GestureClassify

app = Flask(__name__)
app.secret_key = "sinter"
app.debug = True


@app.route("/decode_video", methods=["POST"])
def decode_video():
    result = {
        "status": 200,
        "msg": "video decode was done!"
    }
    video_paths = request.json.get("video_paths")
    out_root = request.json.get("out_root")
    interval = request.json.get("interval")

    # region = request.json.get("region")
    if video_paths:
        for p in tqdm.tqdm(video_paths):
            video_path = p.split("-")[0]
            video_id = p.split("-")[-1]
            p_out_root = os.path.join(out_root, video_id)
            if not os.path.exists(p_out_root):
                os.makedirs(p_out_root)
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)  # 获取fps
            # frame_all = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 获取视频总帧数
            # video_time = frame_all / fps  # 获取视频总时长
            rval, frame = cap.read()
            h, w, _ = frame.shape
            count = 0
            while rval:
                count += 1
                if count % interval != 0:
                    continue
                rval, frame = cap.read()
                if frame is not None:
                    name = str(uuid.uuid1()) + ".jpg"
                    cv2.imwrite(os.path.join(p_out_root, name), frame)
    else:
        result["status"] = 1
        result["msg"] = "没有传入视频文件"
    
    return jsonify(result)


@app.route("/generate_anno", methods=["POST"])
def generate_anno():
    algo_type = request.json.get("algo_type")
    video_paths = request.json.get("video_paths")
    image_paths = request.json.get("image_paths")
    out_root = request.json.get("out_root")
    # is_crop = request.json.get("is_crop")
    # region = request.json.get("region")
    if video_paths:
        for p in video_paths:
            module_managers[algo_type].video_demo(p, out_root, is_save=True)

    if image_paths:
        # print(type(image_paths))
        # print(type(image_paths[0]))
        im_paths = []
        for p in image_paths:
            im_paths += eval(p)
        for p in im_paths:
            module_managers[algo_type].image_demo(p, out_root, is_save=True)

    result = {
        "status": 200,
        "msg": "hello"
    }
    return jsonify(result)


if __name__ == "__main__":

    module_managers = {
        "smoking_calling": GestureClassify(smoking_calling_gesture_params)
    }
    # app.run(host="192.168.31.227", port=9777, debug=True)

    server = pywsgi.WSGIServer(('0.0.0.0', 9777), app)
    server.serve_forever()
