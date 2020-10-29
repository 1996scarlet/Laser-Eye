#!/usr/bin/python3
# -*- coding:utf-8 -*-
from head_pose import get_head_pose
from face_alignment import CoordinateAlignmentModel
from face_detector import MxnetDetectionModel
from gaze_segmentation import MxnetSegmentationModel
import cv2
import numpy as np
from numpy import uint8, frombuffer, prod, zeros, float32, sqrt, arcsin, sin, cos, pi, array, arctan
from numpy.linalg import norm
import time
from queue import Queue
from threading import Thread

SIN_LEFT_THETA = 2 * sin(pi / 4)
SIN_UP_THETA = sin(pi / 6)


def draw_3d_arrow(frame, poi, scale=256):
    right_p0, right_p1, right_iris, right_center = poi

    right_eye_length = norm(right_p0 - right_p1)
    right_ic_distance = norm(right_iris - right_center)
    right_zc_distance = norm(right_iris - right_p0)

    part0 = (right_p0[1] - right_p1[1]) * right_iris[0]
    part1 = (right_p0[0] - right_p1[0]) * right_iris[1]
    part2 = right_p0[0] * right_p1[1]
    part3 = right_p0[1] * right_p1[0]

    right_delta_y = (part0-part1+part2-part3) / right_eye_length / 2
    right_delta_x = sqrt(right_ic_distance**2 - right_delta_y**2)

    right_u_r = right_delta_y * SIN_UP_THETA / right_eye_length
    right_m_r = right_delta_x * SIN_LEFT_THETA / right_eye_length

    theta, pha = arcsin(right_m_r), arcsin(right_u_r)

    # print(f"THETA:{180 * theta / pi}, PHA:{180 * pha / pi}")

    # print(theta, pha)
    # if abs(180 * theta / pi) < 8:
    if abs(theta) < 0.1:
        right_m_r = 0

    if abs(pha) < 0.03:
        right_u_r = 0

    if right_zc_distance**2 - right_delta_y**2 < right_eye_length**2 / 4:
        right_m_r *= -1
        theta *= -1

    cv2.circle(frame, tuple(right_iris.astype(int)), 2, (0, 255, 255), -1)
    cv2.circle(frame, tuple(right_center.astype(int)), 1, (0, 0, 255), -1)

    end = array([right_m_r * scale, right_u_r * scale])

    return theta, pha, end


def get_eye_roi_slice(left_eye_center, right_eye_center):
    '''
    Input:
        Position of left eye, position of right eye.
    Output:
        Eye ROI slice
    Usage: 
        left_slice_h, left_slice_w, right_slice_h, right_slice_w = get_eye_roi_slice(lms[0], lms[1])
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


def main(gpu_ctx=-1):
    cap = cv2.VideoCapture('asset/flame.mp4')
    # cap = cv2.VideoCapture('/home/remilia/ddd.avi')
    # cap = cv2.VideoCapture(2)

    fd = MxnetDetectionModel("weights/16and32", 0, .6, gpu=gpu_ctx)
    fa = CoordinateAlignmentModel('weights/2d106det', 0, gpu=gpu_ctx)
    gs = MxnetSegmentationModel("weights/iris", 0, gpu=gpu_ctx)

    left_eye_bound = [33, 35, 36, 37, 39, 40, 41, 42]
    right_eye_bound = [87, 89, 90, 91, 93, 94, 95, 96]

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        bboxes = fd.detect(frame)

        for landmarks in fa.get_landmarks(frame, bboxes, True):
            # eye ROI slice
            left_slice_h, left_slice_w, right_slice_h, right_slice_w = \
                get_eye_roi_slice(landmarks[34], landmarks[88])

            # eye region of interest
            eyes = (frame[left_slice_h, left_slice_w, :],
                    frame[right_slice_h, right_slice_w, :])

            masks, points = gs.predict(*eyes)

            points += np.array([[left_slice_w.start, left_slice_h.start],
                                [right_slice_w.start, right_slice_h.start]])

            eye_centers = (np.average(landmarks[left_eye_bound], axis=0),
                           np.average(landmarks[right_eye_bound], axis=0))

            _, euler_angle = get_head_pose(landmarks, *frame.shape[:2])
            pitch, yaw, roll = euler_angle[:, 0]

            poi_left = landmarks[35], landmarks[39], points[0], eye_centers[0]
            poi_right = landmarks[89], landmarks[93], points[1], eye_centers[1]

            if yaw > 30:
                theta_left, pha_left, end_mean = draw_3d_arrow(
                    frame, poi_left)
                theta_right, pha_right, _ = draw_3d_arrow(frame, poi_right)
                theta_mean = theta_left
                pha_mean = pha_left * 6
            elif yaw < -30:
                theta_right, pha_right, end_mean = draw_3d_arrow(
                    frame, poi_right)
                theta_left, pha_left, _ = draw_3d_arrow(frame, poi_left)
                theta_mean = theta_right
                pha_mean = pha_right * 6
            else:
                theta_left, pha_left, end_left = draw_3d_arrow(
                    frame, poi_left)
                theta_right, pha_right, end_right = draw_3d_arrow(
                    frame, poi_right)
                end_mean = (end_left + end_right) / 2
                theta_mean = (theta_left + theta_right) / 2
                pha_mean = (pha_left + pha_right) * 3

            zeta = arctan(end_mean[1] / end_mean[0])

            if end_mean[0] < 0:
                zeta += pi

            # print(zeta * 180 / pi)
            # print(zeta)
            if roll < 0:
                roll += 180
            else:
                roll -= 180
            real_angle = zeta + roll * pi / 180
            # print("end mean:", end_mean)
            # print(roll, real_angle * 180 / pi)

            R = sqrt(sum(end_mean ** 2))
            end_mean[0] = R * cos(real_angle)
            end_mean[1] = R * sin(real_angle)

            try:
                cv2.imshow("left", cv2.resize(eyes[0], (480, 240)))
                cv2.imshow("right", cv2.resize(eyes[1], (480, 240)))

                left_eye_hight = landmarks[33, 1] - landmarks[40, 1]
                left_eye_width = landmarks[39, 0] - landmarks[35, 0]

                right_eye_hight = landmarks[87, 1] - landmarks[94, 1]
                right_eye_width = landmarks[93, 0] - landmarks[89, 0]

                for i in landmarks[[left_eye_bound]].astype(int):
                    cv2.circle(frame, tuple(i), 1, (0, 0, 255), -1)

                for i in landmarks[[right_eye_bound]].astype(int):
                    cv2.circle(frame, tuple(i), 1, (0, 0, 255), -1)

                blink_thd = 0.22
                if blink_thd * left_eye_width < left_eye_hight:
                    # cv2.circle(frame, tuple(cp[2].astype(int)), 2, (0, 255, 255), -1)
                    cv2.arrowedLine(frame, tuple(points[0].astype(int)),
                                    tuple((end_mean+points[0]).astype(int)), (0, 125, 255), 2)

                if blink_thd * right_eye_width < right_eye_hight:
                    # cv2.circle(frame, tuple(
                    #     cp[3].astype(int)), 2, (0, 0, 255), -1)
                    cv2.arrowedLine(frame, tuple(points[1].astype(int)),
                                    tuple((end_mean+points[1]).astype(int)), (0, 125, 255), 2)

            except:
                pass

            # frame = fa.draw_poly(frame, landmarks)

        # cv2.imshow('res', cv2.resize(frame, (960, 540)))
        cv2.imshow('res', frame)
        if cv2.waitKey(0) == ord('q'):
            break

    cap.release()


if __name__ == "__main__":
    main()
