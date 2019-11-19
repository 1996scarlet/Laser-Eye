#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import numpy as np
import cv2
import sys
import time
import collections
import mxnet as mx
from numpy import frombuffer, uint8, float32


pred_type = collections.namedtuple('prediction', ['slice', 'close', 'color'])
pred_types = {'face': pred_type(slice(0, 17), False, (173.91, 198.9, 231.795, 0.5)),
              'eyebrow1': pred_type(slice(17, 22), False, (255., 126.99,  14.025, 0.4)),
              'eyebrow2': pred_type(slice(22, 27), False, (255., 126.99,  14.025, 0.4)),
              'nose': pred_type(slice(27, 31), False, (160,  60.945, 112.965, 0.4)),
              'nostril': pred_type(slice(31, 36), False, (160,  60.945, 112.965, 0.4)),
              'eye1': pred_type(slice(36, 42), True, (151.98, 223.125, 137.955, 0.3)),
              'eye2': pred_type(slice(42, 48), True, (151.98, 223.125, 137.955, 0.3)),
              'lips': pred_type(slice(48, 60), True, (151.98, 223.125, 137.955, 0.3)),
              'teeth': pred_type(slice(60, 68), True, (151.98, 223.125, 137.955, 0.4))}


class BaseAlignmentorModel:
    def __init__(self, prefix, epoch, shape, gpu=-1, verbose=False):
        self.device = gpu
        self.ctx = mx.cpu() if self.device < 0 else mx.gpu(self.device)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

        self.model = mx.mod.Module(sym, context=self.ctx, label_names=None)
        self.model.bind(
            data_shapes=[('data', shape)], for_training=False)
        self.model.set_params(arg_params, aux_params)

        self.pre_landmarks = None

    def _preprocess(self, x):
        raise NotImplementedError

    def _inference(self, x):
        raise NotImplementedError

    def probability_density_center(self, masks, b=1e-7):
        masks[masks < 0.6] = 0

        # print(np.sum(masks < 0.6, axis=(1, 2)))
        N, H, W = masks.shape

        masks_sum = np.sum(masks, axis=(1, 2))
        masks_sum += b

        x_sum = np.sum(masks, axis=1) @ np.arange(W)
        y_sum = np.sum(masks, axis=2) @ np.arange(H)

        points = np.stack((x_sum, y_sum), axis=1)
        return points/masks_sum.reshape(-1, 1)

    @staticmethod
    def draw_poly(src, landmarks, stroke=1, color=(125, 255, 125), copy=True):
        if copy:
            draw = src.copy()
        else:
            draw = src

        for pred in pred_types.values():
            le = [landmarks[pred.slice].reshape(-1, 1, 2).astype(np.int32)]
            cv2.polylines(draw, le, pred.close, pred.color, thickness=stroke)

        return draw


class MobileAlignmentorModel(BaseAlignmentorModel):
    def __init__(self, prefix, epoch, gpu=-1, verbose=False):
        shape = (1, 3, 64, 64)
        super().__init__(prefix, epoch, shape, gpu, verbose)

    def _preprocess(self, x):
        cropped = cv2.resize(x, (64, 64)) / 128.0
        inp = cropped.transpose(2, 0, 1)[None, ...]
        return mx.ndarray.array(inp)

    def _inference(self, x):
        db = mx.io.DataBatch(data=[x, ])
        self.model.forward(db, is_train=False)
        return self.model.get_outputs()[-1].asnumpy()

    def get_landmarks(self, image, detected_faces=None):
        for img, box in detected_faces:
            H, W, _ = img.shape

            inp = self._preprocess(img)
            out = self._inference(inp)

            preds = out.reshape((2, 68))

            preds[0] *= W
            preds[0] += box[0]
            preds[1] *= H
            preds[1] += box[1]

            yield preds.T

    def align_one(self, frame, b):

        img = frame[int(b[1]):int(b[3]), int(b[0]):int(b[2]), :]
        H, W, _ = img.shape

        inp = self._preprocess(img)
        out = self._inference(inp)

        preds = out.reshape((2, 68)).astype(float32)

        preds[0] *= W
        preds[0] += b[0]
        preds[1] *= H
        preds[1] += b[1]

        return preds.T


class CabAlignmentorModel(BaseAlignmentorModel):
    def __init__(self, prefix, epoch, gpu=-1, verbose=False):
        shape = (1, 3, 128, 128)
        super().__init__(prefix, epoch, shape, gpu, verbose)

    def _preprocess(self, x):
        cropped = cv2.resize(x, (128, 128))[..., ::-1]
        inp = cropped.transpose(2, 0, 1)[None, ...]
        return mx.ndarray.array(inp)

    def _inference(self, x):
        db = mx.io.DataBatch(data=[x, ])
        self.model.forward(db, is_train=False)
        return self.model.get_outputs()[-1][-1].asnumpy()

    def _calibrate(self, pred, thd):
        if self.pre_landmarks is not None:
            for i in range(68):
                if sum(abs(self.pre_landmarks[i] - pred[i]) < thd) != 2:
                    self.pre_landmarks[i] = pred[i]
        else:
            self.pre_landmarks = pred

    def align_one(self, frame, det, calibrate=False):

        img = frame[int(det[1]):int(det[3]), int(det[0]):int(det[2]), :]

        H, W, _ = img.shape
        offset = W / 64.0, H / 64.0, det[0], det[1]

        inp = self._preprocess(img)
        out = self._inference(inp)

        pred = self._calculate_points(out, offset)

        if calibrate:
            self._calibrate(pred, 2)
            return self.pre_landmarks

        return pred

    def get_landmarks(self, image, detected_faces=None):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detect_faces is None the method will also run a face detector.

        Arguments:
            image {numpy.array} -- The input image.

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None}, format: {x1, y1, x2, y2, score})
        """

        for det in detected_faces:
            yield self.align_one(image, det)
            # yield from self._calculate_points(out, offset)

    def _calculate_points(self, heatmaps, offset, method='pdc'):
        """Obtain (x,y) coordinates given a set of N heatmaps. If the center
        and the scale is provided the function will return the points also in
        the original coordinate frame.

        Arguments:
            hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, H, W]

        Keyword Arguments:
            center {torch.tensor} -- the center of the bounding box (default: {None})
            scale {float} -- face scale (default: {None})
        """

        N, H, W = heatmaps.shape

        if method == 'pdc':
            pred = self.probability_density_center(heatmaps)
        else:
            heatline = heatmaps.reshape(N, H * W)
            indexes = np.argmax(heatline, axis=1)
            x, y = indexes % W, indexes // W
            pred = np.stack((x, y), axis=1).astype(np.float)

        pred *= offset[:2]
        pred += offset[-2:]

        return pred


if __name__ == '__main__':

    V_W, V_H, V_C = 640, 480, 3
    BUFFER_SIZE = V_W * V_H * V_C

    read = sys.stdin.buffer.read
    write = sys.stdout.buffer.write

    # fa = CabAlignmentorModel('weights/cab2d', 0, gpu=0)
    fa = MobileAlignmentorModel('weights/alignment', 0, gpu=0)
    align = fa.align_one

    abyss = np.zeros((68, 2), dtype=float32)

    for src in iter(lambda: read(BUFFER_SIZE + 20), b''):
        frame = frombuffer(src[:BUFFER_SIZE],
                           dtype=uint8).reshape(V_H, V_W, V_C)
        det = frombuffer(src[-20:], dtype=float32)

        res = align(frame, det) if det[3] * det[3] > 0 else abyss

        write(frame)
        write(res.copy(order='C'))
