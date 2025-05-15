import os
import face_recognition
import json
import numpy as np

def extract_face_embeddings(image_path):
    """
    Trích xuất embedding vector từ một ảnh khuôn mặt (đã crop)
    """
    img = face_recognition.load_image_file(image_path)
    # Lấy tất cả các embedding trong ảnh (mặc dù bạn chỉ có một khuôn mặt, sẽ có 1 embedding duy nhất)
    face_embeddings = face_recognition.face_encodings(img)
    
    if not face_embeddings:
        print(f"⚠️ Không tìm thấy khuôn mặt trong {image_path}")
        return None
    
    # Trả về embedding đầu tiên trong danh sách (nếu có nhiều khuôn mặt)
    return face_embeddings[0]

def build_face_database(folder_path, save_path='face_db_face_recognition.json'):
    """
    Xây dựng database khuôn mặt từ thư mục ảnh và lưu dưới dạng JSON
    """
    face_db = []

    for person_id in os.listdir(folder_path):
        person_path = os.path.join(folder_path, person_id)
        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            
            # Trích xuất embedding vector từ ảnh
            embedding = extract_face_embeddings(img_path)
            if embedding is None:
                continue

            # Lưu thông tin ID và vector embedding vào danh sách
            face_db.append({
                "id": person_id,
                "vector": embedding.tolist()  # Chuyển từ numpy array sang list để lưu vào JSON
            })

    # Lưu dữ liệu vào file JSON
    with open(save_path, "w") as f:
        json.dump(face_db, f)
    print(f"✅ Đã lưu {len(face_db)} vector khuôn mặt vào {save_path}")

if __name__ == "__main__":
    # Đường dẫn thư mục chứa ảnh khuôn mặt (đã crop)
    folder_path = "faces_dataset"  # Chỉnh lại theo thư mục của bạn
    build_face_database(folder_path)
