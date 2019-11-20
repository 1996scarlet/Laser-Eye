#!/usr/bin/python3
# -*- coding:utf-8 -*-
import sys
from head_pose import get_head_pose, line_pairs, draw_head_pose_box
from face_alignment import MobileAlignmentorModel, CabAlignmentorModel
from face_detector import MxnetDetectionModel
from gaze_segmentation import MxnetSegmentationModel
from functools import partial
import cv2
import numpy as np
from numpy import uint8, frombuffer, prod, zeros, float32, sqrt, arcsin, sin, cos, pi, array, arctan
from numpy.linalg import norm
import time
from queue import Queue
from threading import Thread

SIN_LEFT_THETA = 2 * sin(pi / 4)
SIN_UP_THETA = sin(pi / 6)


def draw_3d_arrow(frame, poi, scale=640):
    right_p0, right_p1, right_iris = poi
    right_center = (right_p0 + right_p1) / 2

    right_eye_length = norm(right_p0-right_p1)
    right_ic_distance = norm(right_iris-right_center)
    right_zc_distance = norm(right_iris-right_p0)

    part0 = (right_p0[1] - right_p1[1]) * right_iris[0]
    part1 = (right_p0[0] - right_p1[0]) * right_iris[1]
    part2 = right_p0[0] * right_p1[1]
    part3 = right_p0[1] * right_p1[0]

    right_delta_y = (part0-part1+part2-part3) / right_eye_length / 2
    right_delta_x = sqrt(right_ic_distance**2 - right_delta_y**2)

    right_u_r = right_delta_y * SIN_UP_THETA / right_eye_length
    right_m_r = right_delta_x * SIN_LEFT_THETA / right_eye_length

    theta = arcsin(right_m_r)
    pha = arcsin(right_u_r)

    # print(f"THETA:{180 * theta / pi}, PHA:{180 * pha / pi}")

    # if abs(180 * theta / pi) < 8:
    if abs(theta) < 0.12:
        right_m_r = 0

    if right_zc_distance**2 - right_delta_y**2 < right_eye_length**2 / 4:
        right_m_r *= -1

    cv2.circle(frame, tuple(right_center.astype(int)), 1, (0, 0, 255), -1)

    end = array([right_m_r * scale, right_u_r * scale])

    return theta, pha, end


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


gs_thd = 0.1


def plot_mask(src, mask, thd=gs_thd, alpha=0.5):
    draw = src.copy()
    mask = np.repeat((mask > thd)[:, :, :], repeats=3, axis=2)
    draw = np.where(mask, draw * (1 - alpha) + 255 * alpha, draw)
    return draw.astype('uint8')


def main():
    # cap = cv2.VideoCapture('/home/remilia/140.114.77.242/Training_Evaluation_Dataset/Training Dataset/001/noglasses/nonsleepyCombination.avi')
    cap = cv2.VideoCapture('asset/flame.mp4')
    # cap = cv2.VideoCapture(2)

    fd = MxnetDetectionModel("weights/16and32", 0, 1., gpu=0, margin=0.15)
    detect_biggest = partial(fd.detect, mode='biggest')
    margin = fd.margin_clip

    fa = MobileAlignmentorModel('weights/alignment', 0, gpu=0)
    # fa = CabAlignmentorModel('weights/cab2d', 0, gpu=0)
    align = fa.align_one
    cp = zeros((4, 2), dtype=float32)

    gs = MxnetSegmentationModel("weights/iris", 0, gpu=0, thd=gs_thd)
    masker = gs.predict

    while True:
        ret, frame = cap.read()

        det = detect_biggest(frame)

        if det is not None:
            res = margin(det)

            landmarks = align(frame, res) if res[3] - res[1] > 0 else None

            if landmarks is not None:
                cp[0] = (landmarks[36] + landmarks[39]) / 2
                cp[1] = (landmarks[42] + landmarks[45]) / 2

                # eye ROI slice
                left_slice_h, left_slice_w, right_slice_h, right_slice_w = \
                    get_eye_roi_slice(cp[0], cp[1])

                # eye region of interest
                eyes = (frame[left_slice_h, left_slice_w, :],
                        frame[right_slice_h, right_slice_w, :])

                blinks, masks, points = masker(*eyes)

                cp[2, :] = eyes[0].shape[1] * points[0, 0] / 96 + \
                    left_slice_w.start, eyes[0].shape[0] * \
                    points[0, 1] / 48 + left_slice_h.start
                cp[3, :] = eyes[1].shape[1] * points[1, 0] / 96 + \
                    right_slice_w.start, eyes[1].shape[0] * \
                    points[1, 1] / 48 + right_slice_h.start

                reprojectdst, euler_angle = get_head_pose(landmarks)
                pitch, yaw, roll = euler_angle[:, 0]

                is_blink = blinks < 100

                poi_left = (landmarks[36], landmarks[39], cp[2])
                poi_right = (landmarks[42], landmarks[45], cp[3])

                if yaw < -30:
                    theta_left, pha_left, end_mean = draw_3d_arrow(
                        frame, poi_left)
                    theta_right, pha_right, _ = draw_3d_arrow(frame, poi_right)
                elif yaw > 30:
                    theta_right, pha_right, end_mean = draw_3d_arrow(
                        frame, poi_right)
                    theta_left, pha_left, _ = draw_3d_arrow(frame, poi_left)
                else:
                    theta_left, pha_left, end_left = draw_3d_arrow(
                        frame, poi_left)
                    theta_right, pha_right, end_right = draw_3d_arrow(
                        frame, poi_right)
                    end_mean = (end_left + end_right) / 2

                zeta = arctan(end_mean[1] / end_mean[0])

                if end_mean[0] < 0:
                    zeta += pi

                # print(zeta * 180 / pi)
                real_angle = zeta + roll * pi / 180
                # print("end mean:", end_mean)
                # print(roll, real_angle * 180 / pi)

                R = sqrt(sum(end_mean ** 2))
                end_mean[0] = R * cos(real_angle)
                end_mean[1] = R * sin(real_angle)

                try:
                    if not is_blink[0]:
                        cv2.arrowedLine(frame, tuple(cp[2].astype(int)),
                                        tuple((end_mean+cp[2]).astype(int)), (0, 125, 255), 2)

                    if not is_blink[1]:
                        cv2.arrowedLine(frame, tuple(cp[3].astype(int)),
                                        tuple((end_mean+cp[3]).astype(int)), (0, 125, 255), 2)

                except:
                    pass

        cv2.imshow('res', cv2.resize(frame, (960, 540)))
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()


if __name__ == "__main__":
    main()
