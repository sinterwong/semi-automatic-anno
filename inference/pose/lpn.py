from ..keypoints import Keypoints


class KeypointsLPN(Keypoints):
    def __init__(self, weights, input_size=..., conf_thres=0.2):
        super().__init__(weights, input_size, conf_thres)
