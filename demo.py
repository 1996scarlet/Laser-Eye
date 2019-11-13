#!/usr/bin/python3
# -*- coding:utf-8 -*-

from threading import Thread
from queue import Queue
import time
from numpy import uint8, frombuffer, prod, zeros, float32
from numpy.linalg import norm
import cv2
import sys
from functools import partial

from face_detector import MxnetDetectionModel
from face_alignment import MobileAlignmentorModel, CabAlignmentorModel
from gaze_segmentation import MxnetSegmentationModel

# ========== CAMERA SUB PROCESS ==========

FRAME_SHAPE = 480, 640, 3
# FRAME_SHAPE = 720, 1280, 3
BUFFER_SIZE = prod(FRAME_SHAPE)

read = sys.stdin.buffer.read
write = sys.stdout.buffer.write

source_queue = Queue(maxsize=2)
result_queue = Queue(maxsize=2)


# ========== LOAD MODELS ==========

def detect():
    fd = MxnetDetectionModel("weights/16and32", 0, 2., gpu=0, margin=0.15)
    detect_biggest = partial(fd.detect, mode='biggest')
    margin = fd.margin_clip

    for src in iter(partial(read, BUFFER_SIZE), b''):
        # st = time.perf_counter()

        frame = frombuffer(src, dtype=uint8).reshape(FRAME_SHAPE)
        det = detect_biggest(frame)

        if det is not None:
            res = margin(det)
            try:
                source_queue.put_nowait((res, frame))
            except:
                pass

        # print('detect:', time.perf_counter() - st)


def alignment():
    fa = MobileAlignmentorModel("weights/alignment", 0, gpu=0)
    # fa = CabAlignmentorModel('weights/cab2d', 0, gpu=0)

    while True:
        det, data = source_queue.get()

        # st = time.perf_counter()
        res = fa.align_one(data, det) if det[3] * det[3] > 0 else None
        # print('alignment:', time.perf_counter() - st)

        try:
            result_queue.put_nowait((res, data))
        except:
            pass
            # print('Stream queue full', file=sys.stderr)


def segment():
    gs = MxnetSegmentationModel("weights/iris", 0, gpu=0)
    masker = gs.predict
    arrow = gs.draw_arrow

    def get_eye_roi_slice(left_eye_xy, right_eye_xy):
        '''
        Input:
            Position of left eye, position of right eye.
        Output:
            Eye ROI slice
        Usage: 
            left_slice_h, left_slice_w, right_slice_h, right_slice_w = get_eye_roi_slice(lms[0], lms[1])
        '''

        eye_bbox_w = norm(left_eye_xy - right_eye_xy) / 2.7

        half_eye_bbox_w = eye_bbox_w
        half_eye_bbox_h = eye_bbox_w * 0.65

        left_eye_xy = left_eye_xy.astype(int)
        right_eye_xy = right_eye_xy.astype(int)

        left_slice_h = slice(int(left_eye_xy[1]-half_eye_bbox_h),
                             int(left_eye_xy[1]+half_eye_bbox_h))
        left_slice_w = slice(int(left_eye_xy[0]-half_eye_bbox_w),
                             int(left_eye_xy[0]+half_eye_bbox_w))

        right_slice_h = slice(int(right_eye_xy[1]-half_eye_bbox_h),
                              int(right_eye_xy[1]+half_eye_bbox_h))
        right_slice_w = slice(int(right_eye_xy[0]-half_eye_bbox_w),
                              int(right_eye_xy[0]+half_eye_bbox_w))

        return left_slice_h, left_slice_w, right_slice_h, right_slice_w

    cp = zeros((4, 2), dtype=float32)
    while True:
        landmarks, frame = result_queue.get()
        # st = time.perf_counter()

        if landmarks is not None:
            cp[0] = landmarks[36:42].sum(axis=0) / 6
            cp[1] = landmarks[42:48].sum(axis=0) / 6

            # eye ROI slice
            left_slice_h, left_slice_w, right_slice_h, right_slice_w = \
                get_eye_roi_slice(cp[0], cp[1])

            # eye region of interest
            eyes = (frame[left_slice_h, left_slice_w, :],
                    frame[right_slice_h, right_slice_w, :])

            try:
                blinks, _, points = masker(*eyes)

                cp[2, :] = eyes[0].shape[1] * points[0, 0] / 96 + \
                    left_slice_w.start, eyes[0].shape[0] * \
                    points[0, 1] / 48 + left_slice_h.start
                cp[3, :] = eyes[1].shape[1] * points[1, 0] / 96 + \
                    right_slice_w.start, eyes[1].shape[0] * \
                    points[1, 1] / 48 + right_slice_h.start

            except:
                pass
                # print('Eyes ROI too small', file=sys.stderr)

        # print('segment:', time.perf_counter() - st)
        write(frame)
        write(landmarks.copy(order='C'))
        write(blinks < 120)
        write(cp)

        # cv2.imshow('res', frame)
        # cv2.waitKey(1)


segment_thread = Thread(target=segment)
segment_thread.start()

alignment_thread = Thread(target=alignment)
alignment_thread.start()

detect_thread = Thread(target=detect)
detect_thread.start()

# gst-launch-1.0 filesrc location=~/Desktop/uber.mp4 ! decodebin name=dec ! videoconvert ! video/x-raw, format=BGR ! fdsink | ./demo.py | ./draw.py
# cmd_in = ['ffmpeg -c:v libx264 -i ~/Desktop/uber.mp4 -f image2pipe -pix_fmt bgr24 -vcodec rawvideo -']
