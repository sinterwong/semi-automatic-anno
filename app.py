from pickle import FALSE
from flask import Flask, jsonify, request
from params import smoking_calling_gesture_params
from module import GestureClassify

app = Flask(__name__)
app.secret_key = "sinter"
app.debug = True


@app.route("/generate_anno", methods=["POST"])
def generate_anno():
    video_paths = request.json.get("video_paths")
    image_paths = request.json.get("image_paths")
    out_root = request.json.get("out_root")
    algo_type = request.json.get("algo_type")
    # is_crop = request.json.get("is_crop")
    # region = request.json.get("region")

    for p in video_paths:
        module_managers[algo_type].video_demo(p, out_root, is_save=True)

    for p in image_paths:
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

    app.run(host="127.0.0.1", port=9777, debug=True)
