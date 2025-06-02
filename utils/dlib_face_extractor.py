import os
import face_recognition
import json
import numpy as np

def extract_face_embeddings(image_path):
    """
    Trích xuất embedding vector từ một ảnh khuôn mặt đã crop sẵn.

    Args:
        image_path (str): Đường dẫn tới file ảnh đầu vào, ảnh đã được crop chỉ chứa 1 khuôn mặt.

    Output:
        numpy.ndarray: Vector embedding khuôn mặt dạng numpy array.
    """
    img = face_recognition.load_image_file(image_path)
    
    # Lấy kích thước ảnh
    h, w = img.shape[:2]
    # Tọa độ bounding box khuôn mặt: (top, right, bottom, left)
    face_location = [(0, w, h, 0)]
    
    # Trích xuất embedding dùng tọa độ khuôn mặt đã biết, bỏ qua detect
    face_embeddings = face_recognition.face_encodings(img, known_face_locations=face_location)
    
    if not face_embeddings:
        print(f"⚠️ Không tìm thấy khuôn mặt trong {image_path}")
        return None
    
    return face_embeddings[0]

def build_face_database(folder_path, save_path='face_db_face_recognition.json'):
    """
    Xây dựng cơ sở dữ liệu khuôn mặt từ thư mục ảnh và lưu dưới dạng file JSON.

    Args:
        folder_path (str): Đường dẫn thư mục chứa các thư mục con, mỗi thư mục con chứa ảnh khuôn mặt của một người.
        save_path (str, optional): Đường dẫn file JSON để lưu dữ liệu khuôn mặt. Mặc định là 'face_db_face_recognition.json'.

    Returns:
        None
    """
    face_db = []

    for person_id in os.listdir(folder_path):
        person_path = os.path.join(folder_path, person_id)
        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            
            embedding = extract_face_embeddings(img_path)
            if embedding is None:
                continue

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
    folder_path = r"F:\VScode_NHD\intrusion_alert\faces_dataset" 
    build_face_database(folder_path)
