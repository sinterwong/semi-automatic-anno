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

        # forward_start = cv2.getTickCount()
        input_feed = self._get_input_feed(self.input_name, data)
        out = self.session.run(self.output_name, input_feed=input_feed)[
            0]  # 1x512
        # forward_end = cv2.getTickCount()
        # print("推理耗时：{}s".format((forward_end - forward_start) / cv2.getTickFrequency()))
        return out[0]


if __name__ == "__main__":
    weights = "/home/wangjq/wangxt/workspace/pyonnx-example/weights/hand-recognition_0.994.onnx"
    im_path = "/home/wangjq/wangxt/workspace/pyonnx-example/inference/000.jpg"

    feature = Feature(weights, input_size=(64, 64))
    img = cv2.imread(im_path)[:, :, ::-1]

    out = feature.forward(img)
    # 测速
    # imgs = [img] * 100
    # e1 = cv2.getTickCount()
    # for im in imgs:
    #     out = detector.forward(img)
    # e2 = cv2.getTickCount()
    # time = (e2 - e1) / cv2.getTickFrequency()
    # # 关闭视频文件
    # print("总耗时：{}s".format(time))
    # print("单帧耗时：{}s".format(time / 100.))
