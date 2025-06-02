# face_aligner.py

import cv2
import numpy as np

class FaceAligner:
    def __init__(self, output_size=224):
        self.output_size = output_size

    def align(self, image, bbox, face_landmarks):
        """
        Căn chỉnh khuôn mặt trong ảnh dựa trên tọa độ landmark và bounding box.

        Hàm này thực hiện việc:
        - Chuyển các điểm landmark normalized sang tọa độ pixel thực tế.
        - Tính tâm và góc giữa hai mắt để xoay ảnh sao cho mắt nằm ngang.
        - Scale và dịch ảnh sao cho khuôn mặt được căn chỉnh chuẩn về vị trí và kích thước cố định.
        - Crop và resize ảnh khuôn mặt đã căn chỉnh về kích thước đầu ra mong muốn.

        Args:
            image (numpy.ndarray): Ảnh đầu vào (BGR).
            bbox (list hoặc tuple): Bounding box khuôn mặt [l, t, r, b] (dạng pixel).
            face_landmarks (object): Đối tượng landmark khuôn mặt chứa các điểm chuẩn normalized (0..1).

        Returns:
            numpy.ndarray: Ảnh khuôn mặt đã được căn chỉnh và resize về kích thước chuẩn (ví dụ 112x112).
        """
        iw, ih = image.shape[1], image.shape[0]  # Lấy kích thước ảnh (width, height)

        # Chuyển các điểm landmark normalized (0..1) sang tọa độ pixel thực tế trên ảnh
        landmarks_global = np.array([
            [lm.x * iw, lm.y * ih] for lm in face_landmarks.landmark
        ], dtype=np.float32)

        # Tính tâm mắt trái bằng cách lấy trung bình 2 điểm landmark quanh mắt trái (33 và 133)
        left_eye = np.array([
            ((face_landmarks.landmark[33].x + face_landmarks.landmark[133].x) / 2) * iw,
            ((face_landmarks.landmark[33].y + face_landmarks.landmark[133].y) / 2) * ih
        ], dtype=np.float32)

        # Tính tâm mắt phải tương tự, dùng landmark 362 và 263
        right_eye = np.array([
            ((face_landmarks.landmark[362].x + face_landmarks.landmark[263].x) / 2) * iw,
            ((face_landmarks.landmark[362].y + face_landmarks.landmark[263].y) / 2) * ih
        ], dtype=np.float32)

        # Tính trung điểm giữa 2 mắt
        eyes_center = (left_eye + right_eye) / 2

        # Tính độ lệch y và x giữa 2 mắt (dùng để xác định góc quay)
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]

        # Góc quay (theo đơn vị độ), quay ảnh sao cho mắt nằm ngang
        angle = np.degrees(np.arctan2(dY, dX))

        # Khoảng cách giữa 2 mắt (dùng để scale ảnh)
        dist = np.sqrt((dX ** 2) + (dY ** 2))

        # Vị trí mong muốn của mắt phải trên ảnh chuẩn hóa (1.0 - 0.35 = 0.65)
        desiredRightEyeX = 1.0 - 0.35

        # Khoảng cách mong muốn giữa 2 mắt trên ảnh đầu ra
        desiredDist = self.output_size * (desiredRightEyeX - 0.35)

        # Tỉ lệ scale cần để đưa khoảng cách mắt hiện tại về khoảng cách mong muốn
        scale = desiredDist / dist

        # Ma trận affine quay ảnh quanh tâm mắt với góc angle và tỉ lệ scale
        rotation_matrix = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale)

        # Tọa độ đích mà mắt trung tâm sẽ được dịch chuyển tới trong ảnh đầu ra
        tX = self.output_size * 0.5
        tY = self.output_size * 0.4

        # Điều chỉnh ma trận affine để dịch chuyển điểm trung tâm mắt về đúng vị trí đích
        rotation_matrix[0, 2] += tX - eyes_center[0]
        rotation_matrix[1, 2] += tY - eyes_center[1]

        # Chuyển landmark sang dạng tọa độ đồng nhất để nhân với ma trận affine
        landmarks_homo = np.hstack([
            landmarks_global,
            np.ones((landmarks_global.shape[0], 1), dtype=np.float32)
        ])

        # Áp dụng biến đổi affine cho các điểm landmark
        transformed_landmarks = landmarks_homo @ rotation_matrix.T

        # Tính lại bounding box mới trên ảnh đã xoay dựa vào vị trí các landmark sau khi biến đổi
        x_min_new = np.min(transformed_landmarks[:, 0])
        y_min_new = np.min(transformed_landmarks[:, 1])
        x_max_new = np.max(transformed_landmarks[:, 0])
        y_max_new = np.max(transformed_landmarks[:, 1])

        # Xoay toàn bộ ảnh theo ma trận affine, kích thước ảnh đầu ra cố định
        aligned_image = cv2.warpAffine(image, rotation_matrix, (self.output_size, self.output_size), flags=cv2.INTER_CUBIC)

        # Giới hạn vùng crop trong ảnh đầu ra
        x_min_crop = max(0, int(x_min_new))
        y_min_crop = max(0, int(y_min_new))
        x_max_crop = min(self.output_size, int(x_max_new))
        y_max_crop = min(self.output_size, int(y_max_new))

        # Crop vùng khuôn mặt dựa trên bounding box đã tính
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
            continue

        person_img = frame[y1:y2, x1:x2]
        if person_img is None or person_img.size == 0:
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