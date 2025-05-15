import face_recognition
import numpy as np
import json
class FaceRecognitionModule:
    def __init__(self, db_path: str):
        pass
        # self.face_db = self._load_face_database(db_path)

    def _load_face_database(self, json_path: str) -> list:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data

    def _extract_embedding(self, img_path: str) -> np.ndarray:
        """
        Trích xuất embedding vector từ ảnh.
        """
        img = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(img)
        if not encodings:
            return None
        return encodings[0]

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def find_best_match(self, query_vector: np.ndarray, threshold: float = 0.5) -> tuple:
        best_id = 'unknown'
        best_score = -1.0
        for entry in self.face_db:
            db_vector = np.array(entry['vector'], dtype=np.float32)
            score = self._cosine_similarity(query_vector, db_vector)
            if score > best_score:
                best_score = score
                best_id = entry['id']
        if best_score >= threshold:
            return best_id, best_score
        return 'unknown', best_score

    def recognize(self, img_path: str, threshold: float = 0.5) -> tuple:
        query_vector = self._extract_embedding(img_path)
        if query_vector is None:
            return 'no_face_detected', 0.0
        return self.find_best_match(query_vector, threshold)
