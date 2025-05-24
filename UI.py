import cv2
import numpy as np

# --- Thông số ---
UI_SCALE = 1.65
video_h, video_w = int(480 * UI_SCALE), int(680 * UI_SCALE)
face_size = int(96 * UI_SCALE)
info_w = int(160 * UI_SCALE)
num_faces = 5
separator_thickness = 2  # px

def scaled(x): return int(x * UI_SCALE)

# 1. Frame video giả lập
video_frame = np.ones((video_h, video_w, 3), dtype=np.uint8) * 70

# 2. Ảnh khuôn mặt (giả)
face_images = []
for _ in range(num_faces):
    face = np.random.randint(0, 255, (face_size, face_size, 3), dtype=np.uint8)
    face_images.append(face)
    face_images.append(np.zeros((separator_thickness, face_size, 3), dtype=np.uint8))  # separator

# Xóa separator cuối nếu có
face_images = face_images[:-1]
face_column = np.vstack(face_images)

# 3. Thông tin nhận dạng
infos = [("Ellie", "ID001", 0.95),
         ("Kiana", "ID002", 0.92),
         ("Seele", "ID003", 0.87),
         ("Unknow", "ID004", 0.90),
         ("Unknown", "ID005", 0.89)]

info_blocks = []
for name, uid, conf in infos:
    block = np.ones((face_size, info_w, 3), dtype=np.uint8) * 240
    cv2.putText(block, name, (scaled(10), scaled(28)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * UI_SCALE, (0, 0, 0), 1)
    cv2.putText(block, f"ID: {uid}", (scaled(10), scaled(50)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * UI_SCALE, (0, 0, 0), 1)
    cv2.putText(block, f"Conf: {conf:.2f}", (scaled(10), scaled(70)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * UI_SCALE, (0, 0, 0), 1)
    info_blocks.append(block)
    info_blocks.append(np.zeros((separator_thickness, info_w, 3), dtype=np.uint8))  # separator

# Xóa separator cuối nếu có
info_blocks = info_blocks[:-1]
info_column = np.vstack(info_blocks)

# 4. Resize cột nhỏ cho khớp chiều cao
target_height = video_frame.shape[0]
face_column = cv2.resize(face_column, (face_size, target_height))
info_column = cv2.resize(info_column, (info_w, target_height))

# 5. Ghép giao diện cuối cùng
final_display = np.hstack((video_frame, face_column, info_column))

# Hiển thị
cv2.imshow("Surveillance UI (with dividers)", final_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
