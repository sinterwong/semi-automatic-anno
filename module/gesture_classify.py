import os
import numpy as np
import cv2
from inference import DetectorYolov5, Classifier
import urllib
from .base import ModuleBase


class GestureClassify(ModuleBase):
    def __init__(self, params: dict):
        super().__init__(params)
        # load detection model
        self._detector = DetectorYolov5(
            params["det_model_path"], input_size=params["det_input_size"], conf_thres=params["det_conf_thr"], iou_thres=params["det_iou_thr"])

        # load classifier model
        self._classifier = Classifier(
            params["cls_model_path"], input_size=params["cls_input_size"])

        # idx -> classes
        self.idx2classes = params["idx2classes"]

    def _single_frame(self, img, is_save=False):
        out = self._detector.forward(img)
        if out.shape[0] < 1:
            return None
        categorys = []
        objs = []
        for _, dr in enumerate(out):
            hand_image = img[dr[1]: dr[3], dr[0]: dr[2], :][:, :, ::-1]
            if hand_image.shape[0] < 10 or hand_image.shape[1] < 10:
                continue
            class_id = self._classifier.forward(hand_image)
            if is_save:
                categorys.append((class_id, hand_image))
            else:
                categorys.append(class_id)
            objs.append(dr)
        return objs, categorys

    def _visual(self, frame, objs, categorys, thickness=1):
        if objs:
            for i, dr in enumerate(objs):
                cv2.rectangle(frame, (dr[0], dr[1]),
                              (dr[2], dr[3]), (0, 0, 255), 3, 1)
                cv2.putText(frame, self.idx2classes[categorys[i]], (
                    dr[0], dr[1] + (dr[3] - dr[1]) // 2), cv2.FONT_HERSHEY_COMPLEX, thickness, (0, 0, 255), 1)
        return frame

    def video_demo(self, video_file, out_root=None, is_show=False, is_save=False):
        if out_root and not os.path.exists(out_root):
            os.makedirs(out_root)
        if video_file == "0":
            video_file = 0
        frame_iter = self._video(video_file)
        fps, h, w = next(frame_iter)
        # self._video 之后生成 self.ofps, self.ow, self.oh
        run_count = 0
        while True:
            run_count += 1
            try:
                frame = next(frame_iter)
                # 获取检测和关键点推理的结果
                out = self._single_frame(frame[:, :, ::-1], is_save)
                if out is None:
                    continue
                objs, categorys = out
                if len(objs) < 1:
                    continue
                if is_save:
                    for i, im in enumerate(categorys):
                        class_name = self.idx2classes[im[0]]
                        if not os.path.exists(os.path.join(out_root, class_name)):
                            os.makedirs(os.path.join(out_root, class_name))
                        cv2.imwrite(os.path.join(out_root, class_name, "%d_%d_%03d_%s.jpg" % (
                            im[0], run_count, i, os.path.basename(video_file).split(".")[0])), im[1])
                # 可视化结果
                if is_show:
                    frame = self._visual(
                        frame, objs=objs, categorys=list(zip(*categorys))[0])
                    cv2.imshow("demo", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("I'm done!")
                        break
            except StopIteration as e:
                print('Done!')
                break

    def image_demo(self, path, out_root=None, is_show=False, is_save=False):
        if out_root and not os.path.exists(out_root):
            os.makedirs(out_root)
        if path.split(":")[0] in ["http", "https"]:
            with urllib.request.urlopen(path) as url:
                resp = url.read()
                frame = np.asarray(bytearray(resp), dtype="uint8")
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        else:
            frame = cv2.imread(path)
        if not isinstance(frame, np.ndarray):
            return None
        out = self._single_frame(frame[:, :, ::-1], is_save)
        if out is None:
            return None
        objs, categorys = out
        if len(objs) < 1:
            return None
        if is_save:
            for i, im in enumerate(categorys):
                class_name = self.idx2classes[im[0]]
                if not os.path.exists(os.path.join(out_root, class_name)):
                    os.makedirs(os.path.join(out_root, class_name))
                cv2.imwrite(os.path.join(out_root, class_name, "%d_%03d_%s.jpg" % (
                    im[0], i, os.path.basename(path).split(".")[0])), im[1])
        # 可视化结果
        if is_show:
            frame = self._visual(frame, objs=objs, categorys=categorys)
            cv2.imshow("demo", frame)
            cv2.waitKey(2000)
