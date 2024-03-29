from pathlib import Path
import os
import cv2
import numpy as np
from .base import ONNXBase


class Detector(ONNXBase):
    def __init__(self, weights, input_size=(640, 640), conf_thres=0.2, iou_thres=0.45):
        super().__init__(weights, input_size)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def xywh2xyxy(self, x, padw=32, padh=32):
        # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = (x[:, 0] - x[:, 2] / 2)  # top left x
        y[:, 1] = (x[:, 1] - x[:, 3] / 2)  # top left y
        y[:, 2] = (x[:, 0] + x[:, 2] / 2)  # bottom right x
        y[:, 3] = (x[:, 1] + x[:, 3] / 2)  # bottom right y
        return y

    def box_area(self, box):
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

    def _nms(self, boxes, scores, iou_thres):
        keep_index = []
        order = scores.argsort()[::-1]
        areas = self.box_area(boxes)

        while order.size > 1:
            i = order[0]  # 取最大得分
            keep_index.append(i)

            # 计算最高得分与剩余矩形框的相较区域
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter) + 1e-8

            # 保留小于阈值的box
            inds = np.where(iou <= iou_thres)[0]

            # 注意这里索引加了1,因为iou数组的长度比order数组的长度少一个
            order = order[inds + 1]
        return np.array(keep_index, dtype=np.int32)

    def non_max_suppression(self, prediction, conf_thres=0.2, iou_thres=0.45, convert=True, agnostic=False, labels=()):
        """Performs Non-Maximum Suppression (NMS) on inference results

        Returns:
            detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
        """
        xc = prediction[:, :, 4] > conf_thres
        nc = prediction.shape[2] - 5

        multi_label = nc > 1

        max_wh = 4096
        max_det = 300
        output = [np.zeros((0, 6))] * prediction.shape[0]
        for xi, x in enumerate(prediction):
            x = x[xc[xi]]
            if not x.shape[0]:
                continue
            x[:, 5:] *= x[:, 4:5]
            
            if convert:
                box = self.xywh2xyxy(x[:, :4])
            else:
                box = x[:, :4]

            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero()
                x = np.concatenate((box[i], x[i, j + 5, None], j[:, None]), 1)
            else:
                conf = x[:, 5:].max(1, keepdims=True)
                j = np.zeros_like(conf)
                x = np.concatenate((box, conf, j), 1)[
                    conf.reshape(-1) > conf_thres]
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # 区别每个类别的框, 达成同时做NMS
            # boxes (offset by class), scores
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = self._nms(boxes, scores, iou_thres)
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            output[xi] = x[i]
        return output
