import cv2
import mediapipe as mp
from config.cfg_py import config
import numpy as np

class FaceDetectionPredictor:
    def __init__(self):
        detection_conf = config.get("face_detection_model.min_detection_confidence", 0.5)
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.output_size = 224
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=detection_conf,
            refine_landmarks=False 
        )

    def process(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mesh_results = self.face_mesh.process(image_rgb)

        ih, iw, _ = image.shape

        faces = []  # Danh sách mặt, mỗi mặt là dict chứa bbox + landmarks cần align

        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                x_min, y_min = iw, ih
                x_max, y_max = 0, 0

                # Tính bounding box
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * iw), int(landmark.y * ih)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                # Vẽ bounding box
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Vẽ landmarks
                self.mp_drawing.draw_landmarks(
                    image,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1),
                )

                faces.append({
                    "bbox": (x_min, y_min, x_max, y_max),
                    "landmarks":face_landmarks,
                })

        return image, faces

    def align_face(self, image, bbox, face_landmarks):
        # image: ảnh gốc BGR
        # bbox: (x_min, y_min, x_max, y_max)
        # landmarks: dict với left_eye, right_eye đã được tịnh tiến về bbox

        x_min, y_min, x_max, y_max = bbox
        iw, ih = image.shape[1], image.shape[0]

        # Convert normalized landmarks → pixel coordinates (global image)
        landmarks_global = np.array([
            [lm.x * iw, lm.y * ih] for lm in face_landmarks.landmark
        ], dtype=np.float32)

        left_eye = np.array([
            ((face_landmarks.landmark[33].x + face_landmarks.landmark[133].x) / 2) * iw,
            ((face_landmarks.landmark[33].y + face_landmarks.landmark[133].y) / 2) * ih
        ], dtype=np.float32)

        right_eye = np.array([
            ((face_landmarks.landmark[362].x + face_landmarks.landmark[263].x) / 2) * iw,
            ((face_landmarks.landmark[362].y + face_landmarks.landmark[263].y) / 2) * ih
        ], dtype=np.float32)

        
        # # Dùng 2 mắt để tính affine transform như trước
        # left_eye = np.array([face_landmarks.landmark[468].x * iw,
        #                     face_landmarks.landmark[468].y * ih], dtype=np.float32)
        # right_eye = np.array([face_landmarks.landmark[473].x * iw,
        #                     face_landmarks.landmark[473].y * ih], dtype=np.float32)

        eyes_center = (left_eye + right_eye) / 2
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Scale
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredRightEyeX = 1.0 - 0.35
        desiredDist = self.output_size * (desiredRightEyeX - 0.35)
        scale = desiredDist / dist

        # Affine transform
        rotation_matrix = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale)
        tX = self.output_size * 0.5
        tY = self.output_size * 0.4
        rotation_matrix[0, 2] += tX - eyes_center[0]
        rotation_matrix[1, 2] += tY - eyes_center[1]

        # Áp dụng transform lên tất cả 468 landmarks
        landmarks_homo = np.hstack([
            landmarks_global,
            np.ones((landmarks_global.shape[0], 1), dtype=np.float32)
        ])

        transformed_landmarks = landmarks_homo @ rotation_matrix.T  # [468, 2]

        # Tính bounding box mới từ landmarks đã transform
        x_min_new = np.min(transformed_landmarks[:, 0])
        y_min_new = np.min(transformed_landmarks[:, 1])
        x_max_new = np.max(transformed_landmarks[:, 0])
        y_max_new = np.max(transformed_landmarks[:, 1])

        # Crop từ ảnh đã transform
        aligned_image = cv2.warpAffine(image, rotation_matrix, (self.output_size, self.output_size), flags=cv2.INTER_CUBIC)

        # Clamp to valid region và cast về int
        x_min_crop = max(0, int(x_min_new))
        y_min_crop = max(0, int(y_min_new))
        x_max_crop = min(self.output_size, int(x_max_new))
        y_max_crop = min(self.output_size, int(y_max_new))

        cropped_face = aligned_image[y_min_crop:y_max_crop, x_min_crop:x_max_crop]

        # Resize về kích thước chuẩn 224x224
        final_aligned = cv2.resize(cropped_face, (self.output_size, self.output_size), interpolation=cv2.INTER_CUBIC)

        return final_aligned

    def save(self, image, path='output.png'):
        cv2.imwrite(path, image)

    def release(self):
        self.face_mesh.close()

