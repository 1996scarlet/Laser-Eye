#!/usr/bin/python3
# -*- coding:utf-8 -*-

from head_pose import get_head_pose
from face_alignment import CoordinateAlignmentModel
from face_detector import MxnetDetectionModel
from gaze_segmentation import MxnetSegmentationModel
import cv2
import numpy as np
from numpy import sin, cos, pi, arctan
from numpy.linalg import norm
import time
from queue import Queue
from threading import Thread
import sys

SIN_LEFT_THETA = 2 * sin(pi / 4)
SIN_UP_THETA = sin(pi / 6)


def calculate_3d_gaze(frame, poi, scale=256):
    starts, ends, pupils, centers = poi

    eye_length = norm(starts - ends, axis=1)
    ic_distance = norm(pupils - centers, axis=1)
    zc_distance = norm(pupils - starts, axis=1)

    s0 = (starts[:, 1] - ends[:, 1]) * pupils[:, 0]
    s1 = (starts[:, 0] - ends[:, 0]) * pupils[:, 1]
    s2 = starts[:, 0] * ends[:, 1]
    s3 = starts[:, 1] * ends[:, 0]

    delta_y = (s0 - s1 + s2 - s3) / eye_length / 2
    delta_x = np.sqrt(abs(ic_distance**2 - delta_y**2))

    delta = np.array((delta_x * SIN_LEFT_THETA,
                      delta_y * SIN_UP_THETA))
    delta /= eye_length
    theta, pha = np.arcsin(delta)

    # print(f"THETA:{180 * theta / pi}, PHA:{180 * pha / pi}")
    # delta[0, abs(theta) < 0.1] = 0
    # delta[1, abs(pha) < 0.03] = 0

    inv_judge = zc_distance**2 - delta_y**2 < eye_length**2 / 4

    delta[0, inv_judge] *= -1
    theta[inv_judge] *= -1
    delta *= scale

    # cv2.circle(frame, tuple(pupil.astype(int)), 2, (0, 255, 255), -1)
    # cv2.circle(frame, tuple(center.astype(int)), 1, (0, 0, 255), -1)

    return theta, pha, delta.T


def get_eye_roi_slice(left_eye_center, right_eye_center):
    '''
    Input:
        Position of left eye, position of right eye.
    Output:
        Eye ROI slice
    Usage:
        left_slice_h, left_slice_w, right_slice_h, right_slice_w = get_eye_roi_slice(
            lms[0], lms[1])
    '''

    eye_bbox_w = norm(left_eye_center - right_eye_center) / 3.2

    half_eye_bbox_w = eye_bbox_w
    half_eye_bbox_h = eye_bbox_w / 2.0

    left_slice_h = slice(int(left_eye_center[1]-half_eye_bbox_h),
                         int(left_eye_center[1]+half_eye_bbox_h))
    left_slice_w = slice(int(left_eye_center[0]-half_eye_bbox_w),
                         int(left_eye_center[0]+half_eye_bbox_w))

    right_slice_h = slice(int(right_eye_center[1]-half_eye_bbox_h),
                          int(right_eye_center[1]+half_eye_bbox_h))
    right_slice_w = slice(int(right_eye_center[0]-half_eye_bbox_w),
                          int(right_eye_center[0]+half_eye_bbox_w))

    return left_slice_h, left_slice_w, right_slice_h, right_slice_w


def draw(src, blink_thd=0.22, color=(0, 125, 255), copy=False):
    if copy:
        src = src.copy()

    if left_eye_hight / left_eye_width > blink_thd:
        # cv2.circle(frame, tuple(cp[2].astype(int)), 2, (0, 255, 255), -1)
        cv2.arrowedLine(src, tuple(pupils[0].astype(int)),
                        tuple((offset+pupils[0]).astype(int)), color, 2)

    if blink_thd * right_eye_width < right_eye_hight:
        # cv2.circle(frame, tuple(cp[3].astype(int)), 2, (0, 0, 255), -1)
        cv2.arrowedLine(src, tuple(pupils[1].astype(int)),
                        tuple((offset+pupils[1]).astype(int)), color, 2)

    return src


def main(video, gpu_ctx=-1):
    cap = cv2.VideoCapture(video)

    fd = MxnetDetectionModel("weights/16and32", 0, .6, gpu=gpu_ctx)
    fa = CoordinateAlignmentModel('weights/2d106det', 0, gpu=gpu_ctx)
    gs = MxnetSegmentationModel("weights/iris", 0, gpu=gpu_ctx)

    eye_bound = ([33, 35, 36, 37, 39, 40, 41, 42],
                 [87, 89, 90, 91, 93, 94, 95, 96])

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        bboxes = fd.detect(frame)

        for landmarks in fa.get_landmarks(frame, bboxes, calibrate=True):
            # eye ROI slice
            left_slice_h, left_slice_w, right_slice_h, right_slice_w = \
                get_eye_roi_slice(landmarks[34], landmarks[88])

            # eye region of interest
            eyes = (frame[left_slice_h, left_slice_w, :],
                    frame[right_slice_h, right_slice_w, :])

            masks, pupils = gs.predict(*eyes)

            pupils += np.array([[left_slice_w.start, left_slice_h.start],
                                [right_slice_w.start, right_slice_h.start]])

            eye_markers = np.take(landmarks, eye_bound, axis=0)
            eye_centers = np.average(eye_markers, axis=1)

            _, euler_angle = get_head_pose(landmarks, *frame.shape[:2])
            pitch, yaw, roll = euler_angle[:, 0]

            poi = landmarks[[35, 89]], landmarks[[39, 93]], pupils, eye_centers
            theta, pha, delta = calculate_3d_gaze(frame, poi)

            if yaw > 30:
                end_mean = delta[0]
            elif yaw < -30:
                end_mean = delta[1]
            else:
                end_mean = np.average(delta, axis=0)

            if end_mean[0] < 0:
                zeta = arctan(end_mean[1] / end_mean[0]) + pi
            else:
                zeta = arctan(end_mean[1] / (end_mean[0] + 1e-7))

            # print(zeta * 180 / pi)
            # print(zeta)
            if roll < 0:
                roll += 180
            else:
                roll -= 180

            real_angle = zeta + roll * pi / 180
            # print("end mean:", end_mean)
            # print(roll, real_angle * 180 / pi)

            R = norm(end_mean)
            offset = R * cos(real_angle), R * sin(real_angle)

            cv2.imshow("left", cv2.resize(eyes[0], (480, 240)))
            cv2.imshow("right", cv2.resize(eyes[1], (480, 240)))

            left_eye_hight = landmarks[33, 1] - landmarks[40, 1]
            left_eye_width = landmarks[39, 0] - landmarks[35, 0]

            right_eye_hight = landmarks[87, 1] - landmarks[94, 1]
            right_eye_width = landmarks[93, 0] - landmarks[89, 0]

            for i in eye_markers.reshape(-1, 2).astype(int):
                cv2.circle(frame, tuple(i), 1, (0, 0, 255), -1)

            # frame = fa.draw_poly(frame, landmarks)

        # cv2.imshow('res', cv2.resize(frame, (960, 540)))
        cv2.imshow('res', frame)
        if cv2.waitKey(0) == ord('q'):
            break

    cap.release()


if __name__ == "__main__":
    main(sys.argv[1])
