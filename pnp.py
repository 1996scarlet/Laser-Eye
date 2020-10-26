import cv2
import numpy as np
from face_detector import MxnetDetectionModel
from coor_alignment import CoordinateAlignmentModel

dist_coeffs = np.zeors((4, 1), dtype=np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def draw_head_pose_box(frame, reprojectdst, color=(0, 255, 255)):
    for start, end in line_pairs:
        cv2.line(frame, reprojectdst[start], reprojectdst[end], color, 2)


def get_head_pose(shape, H, W):
    cam_matrix = np.array([[W, 0, W/2.0],
                           [0, W, H/2.0],
                           [0, 0, 1]], dtype=np.float32)

    if len(shape) == 68:
        image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                                shape[39], shape[42], shape[45], shape[31], shape[35],
                                shape[48], shape[54], shape[57], shape[8]])
    elif len(shape) == 98:
        image_pts = np.float32([shape[33], shape[38], shape[50], shape[46], shape[60],
                                shape[64], shape[68], shape[72], shape[55], shape[59],
                                shape[76], shape[82], shape[85], shape[16]])
    elif len(shape) == 106:
        image_pts = np.float32([shape[43], shape[50], shape[102], shape[101], shape[35],
                                shape[75], shape[81], shape[93], shape[77], shape[83],
                                shape[52], shape[61], shape[53], shape[0]])
    else:
        raise RuntimeError('Unsupported shape format')

    _, rotation_vec, translation_vec = cv2.solvePnP(
        object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle


def main():
    fd = MxnetDetectionModel("weights/16and32", 0, scale=.4, gpu=-1)
    fa = CoordinateAlignmentModel('weights/2d106det', 0)

    cap = cv2.VideoCapture('asset/flame.mp4')

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        bboxes = fd.detect(frame)

        color = (125, 255, 125)

        for pred in fa.get_landmarks(frame, bboxes, True):
            for p in np.round(pred).astype(np.int):
                cv2.circle(frame, tuple(p), 1, color, 1, cv2.LINE_AA)

            reprojectdst, euler_angle = get_head_pose(pred, *frame.shape[:2])

            for start, end in line_pairs:
                cv2.line(frame, reprojectdst[start],
                         reprojectdst[end], (0, 255, 255), 2)

                # cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.75, (0, 0, 0), thickness=2)
                # cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.75, (0, 0, 0), thickness=2)
                # cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.75, (0, 0, 0), thickness=2)

        # cv2.imshow("result", cv2.resize(frame, (960, 720)))
        cv2.imshow("result", frame)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()
