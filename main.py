# from detectors.face_detector import FaceDetectionPredictor
# import cv2
# def run_face_detection_pipeline(image_path, save_path='result.png'):

#     # Đọc ảnh đầu vào
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Không thể đọc ảnh từ: {image_path}")
#         return

#     # Khởi tạo class xử lý
#     predictor = FaceDetectionPredictor()

#     # Xử lý ảnh
#     output = predictor.process(image)

#     # Hiển thị kết quả
#     cv2.imshow('Face Detection Result', output)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # Lưu kết quả
#     predictor.save(output, save_path)
#     print(f"Đã lưu ảnh sau xử lý tại: {save_path}")

#     # Giải phóng tài nguyên
#     predictor.release()
    
# run_face_detection_pipeline('test2.png', 'output.png')

from services.intrusion_monitor import IntrusionAlertService
import cv2
import numpy as np
import time
from schemas.area import *
from config.cfg_py import config
from database.mongodb_client import Database


video_path = r"F:\VScode_NHD\intrusion_alert\video_test2.mp4"  # Đường dẫn video đầu vào
output_path = "output_video_2.avi"  # Đường dẫn video đầu ra
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

areas = [
    Area(contour=[(100, 200), (300, 200), (300, 400), (100, 400)], area_id="1"),
    Area(contour=[(400, 100), (600, 100), (600, 300), (400, 300)], area_id="2")
]

def main():
        
    Database.initialize(config.get("database.mongo_uri", None), "IntrusionAlert")
    
    service = IntrusionAlertService(areas=areas)

    frame_idx = 0

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        
        if not ret:
            break

        res = service.service_implement(frame, frame_idx)
        processed_frame = res["image"]  # Giả sử image là frame đã vẽ bbox, text, v.v.

        cv2.imshow("Inference", processed_frame)

        out.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_idx += 1
        end_time = time.time()
        print(f"Frame {frame_idx} processed in {end_time - start_time:.2f} seconds")
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

