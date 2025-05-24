import cv2
import mediapipe as mp
from config.cfg_py import *
import numpy as np

class FaceDetectionPredictor:
    def __init__(self):
        detection_conf   = config.get("face_detection_model.min_detection_confidence", 0.5)
        refine_landmarks = config.get("face_detection_model.refine_landmarks", False)
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing   = mp.solutions.drawing_utils
        self.output_size  = 224
        self.face_mesh    = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=detection_conf,
            refine_landmarks        =refine_landmarks
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

    def release(self):
        self.face_mesh.close()

