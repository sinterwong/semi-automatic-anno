from module import DetectClassify
import os
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    """
    Gesture estimation demo
    """

    opt = parser.parse_args()
    print(opt)

    out_root = opt.out_root
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    idx2classes = dict(enumerate(['0', 'smoking', 'calling']))
    # init model
    params = {
        "det_model_path": opt.det_model_path,
        "cls_model_path": opt.cls_model_path,
        "det_input_size": opt.det_input_size,
        "cls_input_size": opt.cls_input_size,
        "idx2classes": idx2classes,
        "det_conf_thr": opt.det_conf_thr,
        "det_iou_thr": opt.det_iou_thr,
    }
    detcls = DetectClassify(params)

    if opt.video_path:
        detcls.video_demo(opt.video_path, opt.out_root,
                           is_show=True, is_save=True)

    if opt.im_path:
        detcls.image_demo(opt.im_path, opt.out_root, is_show=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_model_path", type=str,
                        default="models/handdetn.onnx", help="det onnx model path")
    parser.add_argument("--cls_model_path", type=str,
                        default="models/smoke_phone.onnx", help="cls onnx model path")
    parser.add_argument("--det_input_size", type=tuple,
                        default=(640, 640), help="det input size")
    parser.add_argument("--cls_input_size", type=tuple,
                        default=(128, 128), help="cls input size")
    parser.add_argument("--det_conf_thr", type=float, default=0.4,
                        help="Det threshold that needs to be displayed")
    parser.add_argument("--det_iou_thr", type=float,
                        default=0.4, help="Det threshold that iou")
    parser.add_argument("--video_path", type=str,
                        default="", help="video path")
    parser.add_argument("--im_path", type=str,
                        default="", help="single image path")
    parser.add_argument("--out_root", type=str,
                        default="", help="result output folder")
    main()
