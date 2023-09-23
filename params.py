gesture_params = {
    "det_model_path": "models/handdetn.onnx",
    "cls_model_path": "models/smoke_phone.onnx",
    "det_input_size": (640, 640),
    "cls_input_size": (128, 128),
    "idx2classes": dict(enumerate(['0', '1', '2'])),
    "det_conf_thr": 0.3,
    "det_iou_thr": 0.4
}
