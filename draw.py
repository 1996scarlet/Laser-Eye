#!/usr/bin/python3
# -*- coding:utf-8 -*-

import sys
from cv2 import imshow, imwrite, waitKey, addWeighted, destroyAllWindows, circle, line, arrowedLine, rectangle, fillPoly
from numpy import frombuffer, uint8, float32, prod, zeros, concatenate, array
from numpy import exp, sqrt, arcsin, pi, sin
from functools import partial
from head_pose import get_head_pose, line_pairs, draw_head_pose_box
from numpy.linalg import norm

SIN_LEFT_THETA = 2 * sin(pi / 4)
SIN_UP_THETA = sin(pi / 6)


def draw_2d_arrow(frame, is_blink, points):
    for b, c, p in zip(is_blink, points[:2], points[2:]):
        if not b:
            rw = int(c[0] + 8 * (p[0] - c[0]))
            rh = int(c[1] + 8 * (p[1] - c[1]))
            arrowedLine(frame, tuple(c), (rw, rh), (0, 125, 255), 2)


def draw_3d_arrow(frame, poi, pre, scale=16):
    right_p0, right_p1, right_iris = poi
    right_center = (right_p0 + right_p1) / 2

    if pre is not None:
        if sum(abs(pre[2] - right_center) < 2) == 2:
            right_p0, right_p1, right_center = pre[:3]
        else:
            # print('update pre', right_center)
            pre[:3] = right_p0, right_p1, right_center

        if sum(abs(pre[3] - right_iris) < 2) == 2:
            right_iris = pre[3]
        else:
            # print('update iris', right_iris)
            pre[3] = right_iris
    else:
        pre = [right_p0, right_p1, right_center, right_iris]

    right_eye_length = norm(right_p0-right_p1)
    right_ic_distance = norm(right_iris-right_center)

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

    # print(right_m_r, 180 * theta / pi)

    # circle(frame, tuple(right_p0.astype(int)), 1, (0, 0, 255), -1)
    # circle(frame, tuple(right_p1.astype(int)), 1, (0, 0, 255), -1)
    circle(frame, tuple(right_center.astype(int)), 1, (0, 0, 255), -1)

    right_start = right_iris

    if right_iris[0] < right_center[0]:
        xx = -right_delta_x
    else:
        xx = right_delta_x

    # end = right_start + [xx * scale, right_delta_y * scale]
    end = array([xx * scale, right_delta_y * scale])

    return pre, end


def mesh(frame, landmarks):
    landmarks = landmarks.astype(int)

    eyebrow = landmarks[[0, 36, 37, 38, 39, 27, 42, 43, 44, 45,
                         16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]]

    left_humerus = landmarks[[0, 1, 2, 3, 31, 30,
                              29, 28, 27, 39, 40, 41, 36]]

    right_humerus = landmarks[[27, 28, 29, 30, 35, 13,
                               14, 15, 16, 45, 46, 47, 42]]

    left_cheek = landmarks[[3, 4, 5, 6, 7, 8, 57, 58,
                            59, 48, 49, 50, 51, 33, 32, 31]]

    right_cheek = landmarks[[33, 51, 52, 53, 54, 55, 56,
                             57, 8, 9, 10, 11, 12, 13, 35, 34]]

    for point in landmarks:
        circle(frame, tuple(point), 1, (0, 0, 255), -1)

    fillPoly(frame, [eyebrow, left_humerus, right_humerus,
                     right_cheek, left_cheek], (220, 248, 256))


FRAME_SHAPE = 480, 640, 3
# FRAME_SHAPE = 720, 1280, 3
BUFFER_SIZE = prod(FRAME_SHAPE)

read = sys.stdin.buffer.read

counter = 0
pre_left = None
pre_right = None


for src in iter(partial(read, BUFFER_SIZE + 544 + 2 + 32), b''):
    frame = frombuffer(src[:BUFFER_SIZE], dtype=uint8).reshape(FRAME_SHAPE)
    landmarks = frombuffer(src[BUFFER_SIZE:-34],
                           dtype=float32).reshape(68, 2)
    is_blink = frombuffer(src[-34:-32], dtype=bool)
    points = frombuffer(src[-32:], dtype=float32).reshape(4, 2)

    # base = zeros(FRAME_SHAPE, dtype=uint8)
    # mesh(base, landmarks)
    # frame = addWeighted(frame, 0.7, base, 0.3, 0)

    reprojectdst, euler_angle = get_head_pose(landmarks)
    pitch, yaw, roll = euler_angle[:, 0]

    # draw_2d_arrow(frame, is_blink, points)
    poi_left = (landmarks[36], landmarks[39], points[2])
    poi_right = (landmarks[42], landmarks[45], points[3])

    if yaw < -20:
        pre_left, end_mean = draw_3d_arrow(frame, poi_left, pre_left)
    elif yaw > 20:
        pre_right, end_mean = draw_3d_arrow(frame, poi_right, pre_right)
    else:
        pre_left, end_left = draw_3d_arrow(frame, poi_left, pre_left)
        pre_right, end_right = draw_3d_arrow(frame, poi_right, pre_right)
        end_mean = (end_left + end_right) / 2

    try:
        if not is_blink[0]:
            arrowedLine(frame, tuple(points[2].astype(int)),
                        tuple((end_mean+points[2]).astype(int)), (0, 125, 255), 2)

        if not is_blink[1]:
            arrowedLine(frame, tuple(points[3].astype(int)),
                        tuple((end_mean+points[3]).astype(int)), (0, 125, 255), 2)

    except:
        pass

    # try:
    #     draw_head_pose_box(frame, reprojectdst)
    # except:
    #     pass

    # imwrite(f'./video/img{counter:0>4}.jpg', frame)
    # counter += 1

    imshow('res', frame)
    if waitKey(1) == ord('q'):
        break

destroyAllWindows()
