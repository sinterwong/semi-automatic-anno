from pathlib import Path
import os
import cv2
import numpy as np
from .base import ONNXBase


class Feature(ONNXBase):
    def __init__(self, weights, input_size=(224, 224)):
        super().__init__(weights, input_size)

    def forward(self, image, alpha=114.495, beta=57.63):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # pre_start = cv2.getTickCount()
        data, _, _ = self.preprocessing(
            image, aspect_ratio=False, alpha=alpha, beta=beta)
        # pre_end = cv2.getTickCount()
        # print("前处理耗时：{}s".format((pre_end - pre_start) / cv2.getTickFrequency()))

        forward_start = cv2.getTickCount()
        input_feed = self._get_input_feed(self.input_name, data)
        out = self.session.run(self.output_name, input_feed=input_feed)[
            0]  # 1x512
        forward_end = cv2.getTickCount()
        print("特征提取推理耗时：{}s".format((forward_end - forward_start) / cv2.getTickFrequency()))
        return out[0]
