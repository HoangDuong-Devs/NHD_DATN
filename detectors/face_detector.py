import cv2
import mediapipe as mp
from config.cfg_py import config


class FaceDetectionPredictor:
    def __init__(self):
        detection_conf = config.get("face_detection_model.min_detection_confidence", 0.5)
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        # Khởi tạo FaceMesh với các tham số cấu hình
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=detection_conf)

    def process(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Dự đoán landmarks
        mesh_results = self.face_mesh.process(image_rgb)

        ih, iw, _ = image.shape

        # Vẽ landmarks và bounding box
        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                # Tính toán bounding box từ các landmarks
                x_min, y_min = iw, ih
                x_max, y_max = 0, 0

                # Duyệt qua các landmarks để tính toán min/max cho bounding box
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * iw), int(landmark.y * ih)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                # Vẽ bounding box cho khuôn mặt
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Vẽ landmarks lên khuôn mặt
                self.mp_drawing.draw_landmarks(
                    image,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1),
                )

        return image

    def save(self, image, path='output.png'):
        cv2.imwrite(path, image)

    def release(self):
        self.face_mesh.close()
