from flask import Flask, jsonify, request

app = Flask(__name__)
app.secret_key = "sinter"
app.debug = True


@app.route("/generate_anno", methods=["POST"])
def post_hands_points():
    datas = request.form["datas"]
    algo_type = request.form["algo_type"]
    is_crop = request.form["is_crop"]
    region = request.form["region"]

    result = {
        "status": True,
        "other": "hello"
    }
    return jsonify(result)


if __name__ == "__main__":
    # face feature libraries
    app.run(host="127.0.0.1", port=9777, debug=True)
