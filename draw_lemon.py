import cv2
import numpy as np
import sys
from service.iris_localization import IrisLocalizationModel
from service.head_pose import HeadPoseEstimator
from service.face_alignment import CoordinateAlignmentModel
from service.face_detector import MxnetDetectionModel


def marker_builder(filepath, scale=1.3):

    marker = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    def draw_marker(frame, x, y, rad):

        rad = int(rad * scale)
        length = 1 + (rad << 1)

        resized = cv2.resize(marker, (length, length))

        alpha = (resized[:, :, 3] / 255)[..., None].repeat(3, axis=2)
        beta = 1 - alpha

        mask = (resized[:, :, :3] * alpha).astype(np.uint8)

        hs = slice(y - rad, y + rad + 1)
        ws = slice(x - rad, x + rad + 1)

        frame[hs, ws, :] = (frame[hs, ws, :] * beta).astype(np.uint8)
        frame[hs, ws, :] += mask

    return draw_marker


def main(video, gpu_ctx=-1, yaw_thd=40):
    cap = cv2.VideoCapture(video)
    lemon = marker_builder("asset/lemon.png")

    fd = MxnetDetectionModel("weights/16and32", 0, .6, gpu=gpu_ctx)
    fa = CoordinateAlignmentModel('weights/2d106det', 0, gpu=gpu_ctx)
    gs = IrisLocalizationModel("weights/iris_landmark.tflite")
    hp = HeadPoseEstimator("weights/object_points.npy", cap.get(3), cap.get(4))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        bboxes = fd.detect(frame)

        for landmarks in fa.get_landmarks(frame, bboxes, calibrate=True):
            # calculate head pose
            _, euler_angle = hp.get_head_pose(landmarks)
            pitch, yaw, roll = euler_angle[:, 0]

            eye_centers = landmarks[[34, 88]]

            eye_lengths = np.linalg.norm(
                landmarks[[39, 93]] - landmarks[[35, 89]], axis=1)

            if yaw > -yaw_thd:
                iris_left = gs.get_mesh(frame, eye_lengths[0], eye_centers[0])
                (x, y), rad = gs.draw_pupil(iris_left, frame, thickness=1)
                lemon(frame, x, y, rad)

            if yaw < yaw_thd:
                iris_right = gs.get_mesh(frame, eye_lengths[1], eye_centers[1])
                (x, y), rad = gs.draw_pupil(iris_right, frame, thickness=1)
                lemon(frame, x, y, rad)

        cv2.imshow('res', frame)
        if cv2.waitKey(0) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1])
