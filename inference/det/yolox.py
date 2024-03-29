import numpy as np
import cv2
from ..detector import Detector


class DetectorYolox(Detector):
    def __init__(self, weights, input_size=..., conf_thres=0.2, iou_thres=0.45, agnostic_nms=False):
        super().__init__(weights, input_size, conf_thres, iou_thres)
        self.agnostic_nms = agnostic_nms

    def forward(self, image):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # TODO 以下注释为测速
        # pre_start = cv2.getTickCount()
        data, rw, rh = self.preprocessing(
            image, aspect_ratio=True)  # False, alpha=58.0, beta=114. / 58.
        # pre_end = cv2.getTickCount()
        # print("前处理耗时：{}s".format((pre_end - pre_start) / cv2.getTickFrequency()))

        forward_start = cv2.getTickCount()
        input_feed = self._get_input_feed(self.input_name, data)
        pred = self.session.run(
            self.output_name, input_feed=input_feed)[0]
        forward_end = cv2.getTickCount()
        print("前传耗时：{}s".format((forward_end - forward_start) / cv2.getTickFrequency()))

        out = self.non_max_suppression(
            pred, self.conf_thres, self.iou_thres, False, self.agnostic_nms)[0]
        out[:, 0] = out[:, 0] / rw
        out[:, 1] = out[:, 1] / rh
        out[:, 2] = out[:, 2] / rw
        out[:, 3] = out[:, 3] / rh
        
        # post_end = cv2.getTickCount()
        # print("后处理耗时：{}s".format((post_end - post_start) / cv2.getTickFrequency()))
        out[out[:, 0] < 0] = 0
        out[out[:, 1] < 0] = 0
        out[out[:, 2] < 0] = 0
        out[out[:, 3] < 0] = 0
        return out.astype(np.int32)
