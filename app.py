from flask import Flask, jsonify, request
import itertools
from gevent import pywsgi
from params import smoking_calling_gesture_params
from module import GestureClassify

app = Flask(__name__)
app.secret_key = "sinter"
app.debug = True


@app.route("/decode_video", methods=["POST"])
def decode_video():
    video_paths = request.json.get("video_paths")
    out_root = request.json.get("out_root")
    # region = request.json.get("region")
    if video_paths:
        for p in video_paths:
            pass

    result = {
        "status": 200,
        "msg": "video decode was done!"
    }
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
