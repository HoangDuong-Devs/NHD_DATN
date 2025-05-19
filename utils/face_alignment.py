# face_aligner.py

import cv2
import numpy as np

class FaceAligner:
    def __init__(self, output_size=224):
        self.output_size = output_size

    def align(self, image, bbox, face_landmarks):
        iw, ih = image.shape[1], image.shape[0]

        # Convert normalized landmarks → pixel coordinates
        landmarks_global = np.array([
            [lm.x * iw, lm.y * ih] for lm in face_landmarks.landmark
        ], dtype=np.float32)

        # Estimate eye centers
        left_eye = np.array([
            ((face_landmarks.landmark[33].x + face_landmarks.landmark[133].x) / 2) * iw,
            ((face_landmarks.landmark[33].y + face_landmarks.landmark[133].y) / 2) * ih
        ], dtype=np.float32)

        right_eye = np.array([
            ((face_landmarks.landmark[362].x + face_landmarks.landmark[263].x) / 2) * iw,
            ((face_landmarks.landmark[362].y + face_landmarks.landmark[263].y) / 2) * ih
        ], dtype=np.float32)

        eyes_center = (left_eye + right_eye) / 2
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredRightEyeX = 1.0 - 0.35
        desiredDist = self.output_size * (desiredRightEyeX - 0.35)
        scale = desiredDist / dist

        rotation_matrix = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale)
        tX = self.output_size * 0.5
        tY = self.output_size * 0.4
        rotation_matrix[0, 2] += tX - eyes_center[0]
        rotation_matrix[1, 2] += tY - eyes_center[1]

        landmarks_homo = np.hstack([
            landmarks_global,
            np.ones((landmarks_global.shape[0], 1), dtype=np.float32)
        ])

        transformed_landmarks = landmarks_homo @ rotation_matrix.T
        x_min_new = np.min(transformed_landmarks[:, 0])
        y_min_new = np.min(transformed_landmarks[:, 1])
        x_max_new = np.max(transformed_landmarks[:, 0])
        y_max_new = np.max(transformed_landmarks[:, 1])

        aligned_image = cv2.warpAffine(image, rotation_matrix, (self.output_size, self.output_size), flags=cv2.INTER_CUBIC)

        x_min_crop = max(0, int(x_min_new))
        y_min_crop = max(0, int(y_min_new))
        x_max_crop = min(self.output_size, int(x_max_new))
        y_max_crop = min(self.output_size, int(y_max_new))

        cropped_face = aligned_image[y_min_crop:y_max_crop, x_min_crop:x_max_crop]
        final_aligned = cv2.resize(cropped_face, (self.output_size, self.output_size), interpolation=cv2.INTER_CUBIC)

        return final_aligned
