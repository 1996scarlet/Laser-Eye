import cv2
import numpy as np
import sys
import time


class HeadPoseEstimator:

    def __init__(self, filepath, W, H) -> None:
        _predefined = np.load(filepath, allow_pickle=True)
        self.object_pts, self.r_vec, self.t_vec = _predefined
        self.cam_matrix = np.array([[W, 0, W/2.0],
                                    [0, W, H/2.0],
                                    [0, 0, 1]])

        self.origin_width = 144.76935
        self.origin_height = 139.839

    def get_head_pose(self, shape):
        if len(shape) == 68:
            image_pts = shape
        elif len(shape) == 106:
            image_pts = shape[[
                9, 10, 11, 14, 16, 3, 7, 8, 0,
                24, 23, 19, 32, 30, 27, 26, 25,
                43, 48, 49, 51, 50, 102, 103, 104, 105, 101,
                72, 73, 74, 86, 78, 79, 80, 85, 84,
                35, 41, 42, 39, 37, 36, 89, 95, 96, 93, 91, 90,
                52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55,
                65, 66, 62, 70, 69, 57, 60, 54
            ]]

            # center = image_pts.mean(axis=0)
            # top_center = shape[[49, 104]].mean(axis=0)

            # left_width = -np.linalg.norm(shape[13]- center)
            # right_width = np.linalg.norm(shape[29]- center)
            # top_height = -np.linalg.norm(top_center- center)
            # bottom_height = np.linalg.norm(shape[0]- center)

            # wfactor = self.origin_width / (right_width - left_width)
            # hfactor = self.origin_height / (bottom_height - top_height)
        else:
            raise RuntimeError('Unsupported shape format')

        # start_time = time.perf_counter()

        ret, rotation_vec, translation_vec = cv2.solvePnP(
            self.object_pts,
            image_pts,
            cameraMatrix=self.cam_matrix,
            distCoeffs=None,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        rear_size = 100
        rear_depth = -200
        front_depth = 0

        # left_width *= wfactor
        # right_width *= wfactor
        # bottom_height *= hfactor
        # top_height *= hfactor

        left_width = -75
        top_height = -90
        right_width = 75
        bottom_height = 90

        reprojectsrc = np.float32([#[-rear_size, -rear_size, rear_depth],
                                   #[-rear_size, rear_size, rear_depth],
                                   #[rear_size, rear_size, rear_depth],
                                   #[rear_size, -rear_size, rear_depth],
                                   # -------------------------------------
                                   [left_width, bottom_height, front_depth],
                                   [right_width, bottom_height, front_depth],
                                   [right_width, top_height, front_depth],
                                   [left_width, top_height, front_depth]])

        reprojectdst, _ = cv2.projectPoints(reprojectsrc,
                                            rotation_vec,
                                            translation_vec,
                                            self.cam_matrix,
                                            distCoeffs=None)

        # end_time = time.perf_counter()
        # print(end_time - start_time)

        reprojectdst = reprojectdst.transpose((1,0,2)).astype(np.int32)

        # calc euler angle
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        euler_angle = cv2.decomposeProjectionMatrix(pose_mat)[-1]

        return reprojectdst, euler_angle

    @staticmethod
    def draw_head_pose_box(src, pts, color=(0, 255, 255), thickness=2, copy=False):
        if copy:
            src = src.copy()

        cv2.polylines(src, pts, True, color, thickness)
        # cv2.polylines(src, pts[-4:][None, ...], True, (255, 255, 0), thickness)

        # cv2.line(src, tuple(pts[1]), tuple(pts[6]), color, line_width)
        # cv2.line(src, tuple(pts[2]), tuple(pts[7]), color, line_width)
        # cv2.line(src, tuple(pts[3]), tuple(pts[8]), color, line_width)

        return src


def main(filename):

    cap = cv2.VideoCapture(filename)


    fd = MxnetDetectionModel("../weights/16and32", 0, scale=.6, gpu=-1)
    fa = CoordinateAlignmentModel('../weights/2d106det', 0)
    hp = HeadPoseEstimator("../weights/object_points.npy", cap.get(3), cap.get(4))

    color = (125, 255, 125)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        bboxes = fd.detect(frame)

        for pred in fa.get_landmarks(frame, bboxes, True):
            for p in np.round(pred).astype(np.int):
                cv2.circle(frame, tuple(p), 1, color, 1, cv2.LINE_AA)

            reprojectdst, euler_angle = hp.get_head_pose(pred)
            hp.draw_head_pose_box(frame, reprojectdst)

            cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), thickness=2)
            cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), thickness=2)
            cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), thickness=2)

        cv2.imshow("result", frame)

        if cv2.waitKey(0) == ord('q'):
            break


if __name__ == '__main__':
    from face_detector import MxnetDetectionModel
    from face_alignment import CoordinateAlignmentModel
    import os
    
    os.chdir(os.path.dirname(__file__))
    
    main(sys.argv[1])
