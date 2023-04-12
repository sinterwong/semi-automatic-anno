import os
import cv2
from inference import DetectorYolov5, Feature
from tracker import DeepSort
import numpy as np
import time
from .base import ModuleBase


def compute_color_for_labels(label):
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1+t_size[0]+3, y1+t_size[1]+4), color, -1)
        cv2.putText(
            img, label, (x1, y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


class ObjectCounter(ModuleBase):
    def __init__(self, params: dict):
        super().__init__(params)
        # load detection model
        self._detector = DetectorYolov5(
            params["det_model_path"], input_size=params["det_input_size"], conf_thres=params["det_conf_thr"], iou_thres=params["det_iou_thr"])

        # load feature extractor model
        self._extractor = Feature(
            params["feature_model_path"], input_size=params["feature_input_size"])

        # deepsort tracker
        self._deepSort = DeepSort()

    def _single_frame(self, img):
        """
        :param im0: original image, BGR format
        :return:
        """
        out = self._detector.forward(img)
        if out.shape[0] < 1:
            return np.zeros([0, 5])
        bboxes, features = [], []

        # 对检测的结果进行特征提取
        for _, dr in enumerate(out):
            croped_image = img[dr[1]: dr[3], dr[0]: dr[2], :][:, :, ::-1]
            if croped_image.shape[0] < 10 or croped_image.shape[1] < 10:
                continue
            feature = self._extractor.forward(croped_image)
            bboxes.append(dr)
            features.append(feature)
        # ****************************** deepsort ****************************
        outputs = self._deepSort.update(bboxes, features, img)
        return outputs

    def _visual(self, frame, objs, categorys, thickness=1):
        if objs:
            for i, dr in enumerate(objs):
                cv2.rectangle(frame, (dr[0], dr[1]),
                              (dr[2], dr[3]), (0, 0, 255), 3, 1)
                cv2.putText(frame, self.idx2classes[categorys[i]], (
                    dr[0], dr[1] + (dr[3] - dr[1]) // 2), cv2.FONT_HERSHEY_COMPLEX, thickness, (0, 0, 255), 1)
        return frame

    def video_demo(self, video_file, out_root=None, is_show=False):
        if out_root and not os.path.exists(out_root):
            os.makedirs(out_root)
        if video_file == "0":
            video_file = 0
        frame_iter = self._video(video_file)
        fps, h, w = next(frame_iter)
        # self._video 之后生成 self.ofps, self.ow, self.oh
        run_count = 0

        yolo_time, sort_time, avg_fps = [], [], []
        frame = None
        last_out = None
        while True:
            run_count += 1
            try:
                frame = next(frame_iter)
            except StopIteration as e:
                print('Done!')
                break

            if run_count % 3 != 0:
                outputs = last_out
            # 获取检测和关键点推理的结果
            outputs = self._single_frame(frame[:, :, ::-1])
            last_out = outputs

            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                frame = draw_boxes(frame, bbox_xyxy, identities)  # BGR

                # add FPS information on output video
                text_scale = max(1, frame.shape[1] // 1600)
                cv2.putText(frame, 'frame: %d fps: %.2f ' % (run_count, len(avg_fps) / sum(avg_fps)),
                            (20, 20 + text_scale), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255), thickness=2)

            # 可视化结果
            if is_show:
                cv2.imshow("demo", frame)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    cv2.destroyAllWindows()
                    break

    def image_demo(self, path, out_root=None, is_show=False, is_save=False):
        pass
