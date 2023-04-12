import sys
sys.path.append("C:/Users/Sinter/Workspace/Projects/semi-automatic-anno")
from module import ObjectCounter
import os
import argparse


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():

    opt = parser.parse_args()
    print(opt)

    out_root = opt.out_root
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    # init model
    params = {
        "det_model_path": opt.det_model_path,
        "det_input_size": opt.det_input_size,
        "det_conf_thr": opt.det_conf_thr,
        "det_iou_thr": opt.det_iou_thr,
        "feature_model_path": opt.feature_model_path,
        "feature_input_size": opt.feature_input_size,
        "attention_cls": [0],
    }
    gesture = ObjectCounter(params)

    if opt.video_path:
        gesture.video_demo(opt.video_path, opt.out_root,
                           is_show=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_model_path", type=str,
                        default="models/best_person.onnx", help="det onnx model path")
    parser.add_argument("--feature_model_path", type=str,
                        default="models/osnet_x0_5_imagenet.onnx", help="feature onnx model path")
    parser.add_argument("--det_input_size", type=tuple,
                        default=(640, 640), help="det input size")
    parser.add_argument("--feature_input_size", type=tuple,
                        default=(128, 256), help="feature input size")
    parser.add_argument("--det_conf_thr", type=float, default=0.3,
                        help="Det threshold that needs to be displayed")
    parser.add_argument("--det_iou_thr", type=float,
                        default=0.4, help="Det threshold that iou")
    parser.add_argument("--video_path", type=str,
                        default="data/run_c.mp4", help="video path")
    parser.add_argument("--out_root", type=str,
                        default="data/main_result/tracking", help="result output folder")
    main()
