import face_recognition
import numpy as np
import json
from config.cfg_py import config

class FaceRecognitionModule:
    def __init__(self, db_path=config.get("face_recognition.database_path", "face_db.json")):
        self.face_db = self._load_face_database(db_path)

    def _load_face_database(self, json_path: str) -> list:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data

    def _extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        _Trích xuất vector embedding đặc trưng khuôn mặt từ ảnh đầu vào_

        Args:
            image (_np.ndarray_): _Ảnh đầu vào dưới dạng mảng NumPy_

        Returns:
            _np.ndarray | None_: _Vector embedding đặc trưng khuôn mặt nếu tìm thấy, ngược lại trả về None_
        """
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            return None
        return encodings[0]

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def find_best_match(self, query_vector: np.ndarray, threshold: float = 0.5) -> tuple:
        """
        _Tìm đối tượng khớp tốt nhất trong cơ sở dữ liệu dựa trên vector đặc trưng_

        Args:
            query_vector (_np.ndarray_): _Vector đặc trưng của khuôn mặt cần so sánh_
            threshold (_float_, optional): _Ngưỡng độ tương đồng để chấp nhận kết quả khớp. Mặc định là 0.5_

        Returns:
            _tuple_: _Trả về tuple (id của đối tượng khớp tốt nhất, điểm tương đồng). Nếu không có khớp nào vượt ngưỡng, trả về ('unknown', điểm tốt nhất)_
        """
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

    def recognize(self, image: np.ndarray, threshold: float = 0.5) -> tuple:
        """
        _Nhận diện khuôn mặt từ ảnh (dưới dạng mảng NumPy)_

        Args:
            image (_np.ndarray_): _Ảnh đầu vào chứa khuôn mặt cần nhận diện_
            threshold (_float_, optional): _Ngưỡng độ tương đồng để chấp nhận một kết quả khớp. Mặc định là 0.5_

        Returns:
            _tuple_: _Trả về một tuple gồm (tên hoặc nhãn nhận diện, độ tương đồng). Nếu không nhận diện được thì trả về ('unknown', 0.0)_
        """
        query_vector = self._extract_embedding(image)
        if query_vector is None:
            return 'unknown', 0.0
        return self.find_best_match(query_vector, threshold)
