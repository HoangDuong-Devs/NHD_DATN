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


def extract_aligned_faces_from_people(frame, boxes, ids, face_detector, face_aligner):
    """
    Với mỗi người được detect, crop vùng người → chạy face detection + align → trả về ảnh khuôn mặt align (hoặc None nếu không tìm được).
    
    Args:
        frame (np.ndarray): Ảnh gốc.
        boxes (List[List[int]]): Các bbox của người [x1, y1, x2, y2].
        ids (List[int]): ID của từng người tương ứng.
        face_detector: Đối tượng có hàm `process(image)` trả bbox và landmarks.
        face_aligner: Đối tượng có hàm `align(image, bbox, landmarks)` trả ảnh khuôn mặt đã căn chỉnh.

    Returns:
        List[dict]: Mỗi phần tử dạng {"id": int, "face": np.ndarray or None}
    """
    results = []

    for box, person_id in zip(boxes, ids):
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            print(f"[WARN] Crop box invalid: {box}")
            continue

        person_img = frame[y1:y2, x1:x2]
        if person_img is None or person_img.size == 0:
            print(f"[WARN] Cropped image empty at ID: {id}")
            continue
        person_img_copy = person_img.copy()

        # Chạy face detector trên vùng người
        image_with_landmarks, faces = face_detector.process(person_img)

        if len(faces) == 0:
            results.append({
                "id": person_id,
                "face": None
            })
            continue

        # Lấy khuôn mặt đầu tiên
        face = faces[0]
        bbox = face["bbox"]  # (x_min, y_min, x_max, y_max)
        landmarks = face["landmarks"]

        try:
            aligned_face = face_aligner.align(person_img_copy, bbox, landmarks)
        except:
            aligned_face = None

        results.append({
            "id": person_id,
            "face": aligned_face
        })

    return results