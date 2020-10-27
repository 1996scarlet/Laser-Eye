import cv2
import numpy as np
from face_detector import MxnetDetectionModel
from coor_alignment import CoordinateAlignmentModel
import time

dist_coeffs = np.zeros((4, 1), dtype=np.float32)

object_pts = np.float32([[-73.393524, -29.801432, -47.667534],
                         [-72.77502, -10.949766, -45.909405],
                         [-70.53364,   7.929818, -44.84258],
                         [-66.85006,  26.07428, -43.141113],
                         [-59.790188, 42.56439, -38.6353],
                         [-48.368973, 56.48108, -30.750622],
                         [-34.1211,   67.246994, -18.456453],
                         [-17.87541,  75.05689,  -3.609035],
                         [0.098749, 77.06129,   0.881698],
                         [17.477032, 74.758446, -5.181201],
                         [32.648968, 66.92902, -19.176563],
                         [46.372356, 56.31139, -30.77057],
                         [57.34348,  42.419125, -37.628628],
                         [64.38848,  25.45588, -40.88631],
                         [68.212036,  6.990805, -42.28145],
                         [70.486404, -11.666193, -44.142567],
                         [71.375824, -30.36519, -47.140427],
                         [-61.119408, -49.361603, -14.254422],
                         [-51.287586, -58.769794, -7.268147],
                         [-37.8048,  -61.996155, -0.442051],
                         [-24.022755, -61.033398,  6.606501],
                         [-11.635713, -56.68676,  11.967398],
                         [12.056636, -57.391033, 12.051204],
                         [25.106256, -61.902187,  7.315098],
                         [38.33859, -62.777714,  1.022953],
                         [51.191006, -59.302345, -5.349435],
                         [60.053852, -50.190254, -11.615746],
                         [0.65394, -42.19379,  13.380835],
                         [0.804809, -30.993721, 21.150852],
                         [0.992204, -19.944595, 29.284037],
                         [1.226783, -8.414541, 36.94806],
                         [-14.772472,  2.598255, 20.132004],
                         [-7.180239,  4.751589, 23.536684],
                         [0.55592,   6.5629,   25.944448],
                         [8.272499,  4.661005, 23.695742],
                         [15.214351,  2.643046, 20.858156],
                         [-46.04729, -37.471413, -7.037989],
                         #  [-37.674686, -42.73051,  -3.021217],
                         #  [-27.883856, -42.711517, -1.353629],
                         [-19.648268, -36.75474,   0.111088],
                         #  [-28.272964, -35.134495,  0.147273],
                         #  [-38.082417, -34.919044, -1.476612],
                         [19.265867, -37.032307,  0.665746],
                         #  [27.894192, -43.342445, -0.24766],
                         #  [37.43753, -43.11082,  -1.696435],
                         [45.170807, -38.086514, -4.894163],
                         #  [38.196453, -35.532024, -0.282961],
                         #  [28.76499, -35.484287,  1.172675],
                         # --------------------------------
                         [-28.916267, 28.612717,  2.24031],
                         [-17.533194, 22.172188, 15.934335],
                         [-6.68459,  19.02905,  22.611355],
                         [0.381001, 20.721119, 23.748438],
                         [8.375443, 19.03546,  22.721994],
                         [18.876617, 22.39411,  15.610679],
                         [28.794413, 28.079924,  3.217393],
                         [19.057573, 36.29825,  14.987997],
                         [8.956375, 39.634575, 22.554245],
                         [0.381549, 40.395645, 23.591625],
                         [-7.428895, 39.836407, 22.406107],
                         [-18.160633, 36.6779,   15.121907],
                         #  [-24.37749,  28.67777,   4.785684],
                         #  [-6.897633, 25.475977, 20.893742],
                         #  [0.340663, 26.014269, 22.220478],
                         #  [8.444722, 25.326199, 21.02552],
                         #  [24.474474, 28.323008,  5.712776],
                         #  [8.449166, 30.596216, 20.67149],
                         #  [0.205322, 31.408737, 21.90367],
                         #  [-7.198266, 30.844875, 20.328022]
                         ])

rear_size = 64
rear_depth = 0

front_size = 80
front_depth = 80

reprojectsrc = np.float32([[-rear_size, -rear_size, rear_depth],
                           [-rear_size, rear_size, rear_depth],
                           [rear_size, rear_size, rear_depth],
                           [rear_size, -rear_size, rear_depth],
                           [-rear_size, -rear_size, rear_depth],
                           # -------------------------------------
                           [-front_size, -front_size, front_depth],
                           [-front_size, front_size, front_depth],
                           [front_size, front_size, front_depth],
                           [front_size, -front_size, front_depth],
                           [-front_size, -front_size, front_depth]])


def draw_head_pose_box(image, reprojectdst, color=(0, 255, 255), line_width=2):
    cv2.polylines(image, [reprojectdst], True, color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(reprojectdst[1]), tuple(
        reprojectdst[6]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(reprojectdst[2]), tuple(
        reprojectdst[7]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(reprojectdst[3]), tuple(
        reprojectdst[8]), color, line_width, cv2.LINE_AA)


def get_head_pose(shape, H, W):
    cam_matrix = np.array([[W, 0, W/2.0],
                           [0, W, H/2.0],
                           [0, 0, 1]], dtype=np.float32)

    if len(shape) == 68:
        image_pts = shape
    elif len(shape) == 106:
        image_pts = shape[[9, 11, 12, 14, 16, 3, 5, 7, 0,
                           23, 21, 19, 32, 30, 28, 27, 25,
                           43, 48, 49, 51, 50, 102, 103, 104, 105, 101,
                           72, 73, 74, 86, 78, 79, 80, 85, 84,
                           #    35, 41, 42, 39, 37, 36, 89, 95, 96, 93, 91, 90,
                           35, 39, 89, 93,
                           52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55,
                           #    65, 66, 62, 70, 69, 57, 60, 54
                           ]]
    else:
        raise RuntimeError('Unsupported shape format')

    start_time = time.perf_counter()

    _, rotation_vec, translation_vec = cv2.solvePnP(
        object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc,
                                        rotation_vec, translation_vec,
                                        cam_matrix, dist_coeffs)

    end_time = time.perf_counter()

    print(end_time - start_time)
    reprojectdst = reprojectdst.reshape(-1, 2).astype(np.int32)

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle


def main():
    fd = MxnetDetectionModel("weights/16and32", 0, scale=.4, gpu=-1)
    fa = CoordinateAlignmentModel('weights/2d106det', 0)

    # cap = cv2.VideoCapture('/home/remilia/slice.mp4')
    cap = cv2.VideoCapture('/home/remilia/white.avi')
    # cap = cv2.VideoCapture('asset/flame.mp4')

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        bboxes = fd.detect(frame)

        color = (125, 255, 125)

        for pred in fa.get_landmarks(frame, bboxes, False):
            for p in np.round(pred).astype(np.int):
                cv2.circle(frame, tuple(p), 1, color, 1, cv2.LINE_AA)

            reprojectdst, euler_angle = get_head_pose(pred, *frame.shape[:2])

            draw_head_pose_box(frame, reprojectdst)

            cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, color, thickness=2)
            cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, color, thickness=2)
            cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, color, thickness=2)

        # cv2.imshow("result", cv2.resize(frame, (960, 720)))
        cv2.imshow("result", frame)
        if cv2.waitKey(0) == ord('q'):
            break


if __name__ == '__main__':
    main()
