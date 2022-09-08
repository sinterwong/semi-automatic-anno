import cv2
import numpy as np
from ..keypoints import Keypoints
from tools.pose_transforms import transform_preds


class KeypointsDarkPose(Keypoints):
    def __init__(self, weights, input_size=..., conf_thres=0.2):
        super().__init__(weights, input_size, conf_thres)

    def _taylor(self, hm, coord):
        heatmap_height = hm.shape[0]
        heatmap_width = hm.shape[1]
        px = int(coord[0])
        py = int(coord[1])
        if 1 < px < heatmap_width-2 and 1 < py < heatmap_height-2:
            dx = 0.5 * (hm[py][px+1] - hm[py][px-1])
            dy = 0.5 * (hm[py+1][px] - hm[py-1][px])
            dxx = 0.25 * (hm[py][px+2] - 2 * hm[py][px] + hm[py][px-2])
            dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1]
                          + hm[py-1][px-1])
            dyy = 0.25 * (hm[py+2*1][px] - 2 * hm[py][px] + hm[py-2*1][px])
            derivative = np.matrix([[dx], [dy]])
            hessian = np.matrix([[dxx, dxy], [dxy, dyy]])
            if dxx * dyy - dxy ** 2 != 0:
                hessianinv = hessian.I
                offset = -hessianinv * derivative
                offset = np.squeeze(np.array(offset.T), axis=0)
                coord += offset
        return coord

    def _gaussian_blur(self, hm, kernel):
        border = (kernel - 1) // 2
        batch_size = hm.shape[0]
        num_joints = hm.shape[1]
        height = hm.shape[2]
        width = hm.shape[3]
        for i in range(batch_size):
            for j in range(num_joints):
                origin_max = np.max(hm[i, j])
                dr = np.zeros((height + 2 * border, width + 2 * border))
                dr[border: -border, border: -border] = hm[i, j].copy()
                dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
                hm[i, j] = dr[border: -border, border: -border].copy()
            hm[i, j] *= origin_max / np.max(hm[i, j])
        return hm

    def _get_final_preds(self, batch_heatmaps, center, scale):
        coords, maxvals = self._get_max_preds(batch_heatmaps)
        heatmap_height = batch_heatmaps.shape[2]
        heatmap_width = batch_heatmaps.shape[3]

        # post-processing
        batch_heatmaps = self._gaussian_blur(batch_heatmaps, 11)
        batch_heatmaps = np.maximum(batch_heatmaps, 1e-10)
        batch_heatmaps = np.log(batch_heatmaps)
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                coords[n, p] = self._taylor(batch_heatmaps[n][p], coords[n][p])

        preds = coords.copy()

        # Transform back
        for i in range(coords.shape[0]):
            # print (heatmap_height, heatmap_width)
            preds[i] = transform_preds(
                coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
            )

        return preds, maxvals


if __name__ == "__main__":
    weights = "../weights/hrnet_w32_coco_wholebody_256x192_dark-469327ef_20200922.onnx"
    detector = KeypointsDarkPose(weights)
    im_path = "../data/person/000.jpg"
    img = cv2.imread(im_path)[:, :, ::-1]
    show_img = img[:, :, ::-1].copy()

    imgs = [img] * 100
    e1 = cv2.getTickCount()
    for im in imgs:
        points = detector.forward(im)

    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()
    # 关闭视频文件
    print("总耗时：{}s".format(time))
    print("单帧耗时：{}s".format(time / 100.))

    for i, p in enumerate(points):
        cv2.circle(show_img, tuple(p[:2].astype(np.int32)), 5, (0, 0, 255), 2)
    cv2.imwrite('out2.jpg', show_img)
