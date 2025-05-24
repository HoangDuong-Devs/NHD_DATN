# import cv2
# import numpy as np
# import mediapipe as mp
# from detectors.face_detector import FaceDetectionPredictor
# from utils.face_alignment import FaceAligner

# import cv2


# def main():
#     predictor = FaceDetectionPredictor()
#     aligner = FaceAligner(output_size=150)

#     # Đọc ảnh
#     image = cv2.imread(r"F:\VScode_NHD\intrusion_alert\image.png")
#     print("Kích thước ảnh gốc:", image.shape)
#     h, w, _ = image.shape
#     cropped_height = int((1/2) * h)
#     image = image[0:cropped_height, 0:w]

#     target_width = 224
#     scale_ratio = target_width / image.shape[1]
#     target_height = int(image.shape[0] * scale_ratio)
#     image = cv2.resize(image, (target_width, target_height))
    
#     image_1 = image.copy()
#     if image is None:
#         print("Không đọc được ảnh input.jpg")
#         return

#     # Xử lý ảnh, nhận về ảnh vẽ bbox+landmarks và danh sách mặt
#     image_with_landmarks, faces = predictor.process(image)

#     if len(faces) == 0:
#         print("Không phát hiện được khuôn mặt nào.")
#         return

#     # Lấy khuôn mặt đầu tiên
#     face = faces[0]
#     x_min, y_min, x_max, y_max = face["bbox"]
#     landmarks = face["landmarks"]

#     # Align khuôn mặt dựa trên landmarks đã tịnh tiến trong vùng crop
#     aligned_face = aligner.align(image_1, face["bbox"], landmarks)

#     # Hiển thị ảnh gốc đã vẽ bbox+landmarks
#     cv2.imshow("Original Image with Landmarks", image_with_landmarks)
#     cv2.imwrite("output_image_with_landmarks.jpg", image_with_landmarks)
#     # Hiển thị ảnh khuôn mặt đã align
#     cv2.imshow("Aligned Face", aligned_face)
#     cv2.imwrite("output_aligned_face.jpg", aligned_face)
#     print("Nhấn phím bất kỳ để đóng cửa sổ...")
    
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     predictor.release()

# if __name__ == "__main__":
#     main()

from services.intrusion_monitor import IntrusionAlertService
import cv2
import numpy as np
import time
from schemas.area import *
from config.cfg_py import config
from database.log_results_to_db import LogResults

import torch


video_path = r"F:\VScode_NHD\intrusion_alert\video_test2.mp4"  # Đường dẫn video đầu vào
output_path = "output_video_2.avi"  # Đường dẫn video đầu ra
rtsp_path = r"http://192.168.0.102:4747/video"
cap = cv2.VideoCapture(rtsp_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

areas = [
    Area(contour=[(100, 200), (300, 200), (300, 400), (100, 400)], area_id="1"),
    Area(contour=[(400, 100), (600, 100), (600, 300), (400, 300)], area_id="2")
]

def main():
    log_res = LogResults()
    service = IntrusionAlertService(areas=None)

    frame_idx = 0

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        
        

        res = service.service_implement(frame, frame_idx)
        log_res._intrusion_alert_log(res)
        
        processed_frame = res["image"]  # Giả sử image là frame đã vẽ bbox, text, v.v.
        end_time = time.time()
        
        # print(res["intrusion_results"])
        cv2.imshow("Inference", processed_frame)

        out.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        print(f"Frame {frame_idx} processed in {end_time - start_time:.2f} seconds")
        frame_idx += 1
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

