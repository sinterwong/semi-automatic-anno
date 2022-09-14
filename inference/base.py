import os
import cv2
import numpy as np
import onnxruntime
import abc


class ONNXBase(metaclass=abc.ABCMeta):
    def __init__(self, weights, input_size):
        assert os.path.exists(weights), "model file is not found!"
        self.input_size = input_size
        self.session = None
        self.input_name = None
        self.output_name = None
        self._load_model(weights)

    def _load_model(self, weights):
        self.session = onnxruntime.InferenceSession(weights)
        self.input_name = self._get_input_name(self.session)
        self.output_name = self._get_output_name(self.session)

    def _get_output_name(self, session):
        """
        output_name = session.get_outputs()[0].name
        :param session:
        :return:
        """
        output_name = []
        for node in session.get_outputs():
            output_name.append(node.name)
        return output_name

    def _get_input_name(self, session):
        """
        input_name = session.get_inputs()[0].name
        :param session:
        :return:
        """
        input_name = []
        for node in session.get_inputs():
            input_name.append(node.name)
        return input_name

    def _get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def preprocessing(self, image, aspect_ratio=True, alpha=0.0, beta=0.0):
        rh, rw = None, None
        h, w, _ = image.shape
        if aspect_ratio:
            assert (self.input_size[0] == self.input_size[1])
            data = np.zeros(
                [self.input_size[1], self.input_size[0], 3], dtype=np.float32)
            ration = float(self.input_size[0]) / max(h, w)
            ih, iw = round(h * ration), round(w * ration)
            data[:ih, :iw, :] = cv2.resize(image, (iw, ih)).astype(np.float32)
            rh, rw = ration, ration
        else:
            rw = self.input_size[0] / w
            rh = self.input_size[1] / h
            data = cv2.resize(
                image, (self.input_size[0], self.input_size[1])).astype(np.float32)

        if alpha != 0:
            data /= alpha

        if beta != 0:
            data -= beta
        data = data.transpose([2, 0, 1])
        return np.expand_dims(data, 0), rw, rh

    @abc.abstractmethod
    def forward(self, image):
        pass
