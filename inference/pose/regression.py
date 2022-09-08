from pathlib import Path
import os
import cv2
import numpy as np
from ..keypoints import Keypoints
from tools.pose_transforms import xywh2cs


class KeypointsReg(Keypoints):
    def __init__(self, weights, input_size=..., conf_thres=0.2):
        super().__init__(weights, input_size, conf_thres)

    def preprocessing(self, image):
        data = cv2.resize(
            image, (self.input_size[1], self.input_size[0])).astype(np.float32)
        data -= 128.
        data /= 256.
        data = data.transpose([2, 0, 1])

        return np.expand_dims(data, 0)

    def _get_final_preds(self, output, width, height):
        output = np.squeeze(output)
        x, y = [], []
        for i in range(int(output.shape[0]/2)):
            x.append(output[i*2+0]*float(width))
            y.append(output[i*2+1]*float(height))

        preds = np.array(list(zip(x, y)))

        return preds

    def forward(self, image):
        '''
        :param image: (RGB)(H, W, C)
        :return:
        '''
        h, w, _ = image.shape

        data = self.preprocessing(image)

        # forward_start = cv2.getTickCount()
        input_feed = self._get_input_feed(self.input_name, data)
        outputs = self.session.run(self.output_name, input_feed=input_feed)[0]
        # forward_end = cv2.getTickCount()
        # print("推理耗时：{}s".format((forward_end - forward_start) / cv2.getTickFrequency()))

        # post_start = cv2.getTickCount()
        preds = self._get_final_preds(outputs, w, h)
        # post_end = cv2.getTickCount()
        # print("后处理耗时：{}s".format((post_end - post_start) / cv2.getTickFrequency()))

        return preds
