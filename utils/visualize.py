import numpy as np
import cv2

def visualize_intrusion_interface(result, max_faces=5, mouse_pos=None):
    UI_SCALE = 1.65
    scaled = lambda x: int(x * UI_SCALE)

    original_h, original_w = result["image"].shape[:2]
    video_h, video_w = scaled(480), scaled(680)
    face_size = scaled(96)
    info_w = scaled(160)
    separator_thickness = 2
    separator_color = (50, 50, 50)
    border_color = (0, 200, 0)

    # Load default face image
    default_face_path = r"F:\VScode_NHD\intrusion_alert\images\default.png"
    default_face_img = cv2.imread(default_face_path)
    if default_face_img is None:
        default_face_img = np.full((face_size, face_size, 3), (0, 255, 255), dtype=np.uint8)
    else:
        default_face_img = cv2.resize(default_face_img, (face_size, face_size))

    video_frame = cv2.resize(result["image"], (video_w, video_h))
    objects = result.get("faces", [])

    with_face = sorted([obj for obj in objects if obj.get("face") is not None], key=lambda x: x["id"])
    no_face = sorted([obj for obj in objects if obj.get("face") is None], key=lambda x: x["id"])
    combined = (with_face + no_face)[:max_faces]

    target_height = video_frame.shape[0]
    block_height = target_height // max_faces
    sep = separator_thickness

    face_column = np.ones((target_height, face_size + 4, 3), dtype=np.uint8) * 255
    info_column = np.ones((target_height, info_w + 4, 3), dtype=np.uint8) * 255

    for i in range(max_faces):
        y_start = i * block_height
        y_end = y_start + block_height - sep

        if i < len(combined):
            item = combined[i]

            # Prepare face image
            face_img = item["face"] if item.get("face") is not None else default_face_img
            face_img = cv2.copyMakeBorder(face_img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=border_color)
            face_resized = cv2.resize(face_img, (face_size + 4, y_end - y_start))
            face_column[y_start:y_end, :, :] = face_resized

            # Prepare info block
            block = np.full((y_end - y_start, info_w + 4, 3), 240, dtype=np.uint8)
            cv2.rectangle(block, (0, 0), (info_w + 3, y_end - y_start - 1), border_color, 2)

            uid = str(item["id"])
            name = item.get("name", "Unknown")
            conf = item.get("confidence", -1)
            conf_text = f"{conf:.2f}" if conf is not None and conf >= 0 else "?"

            cv2.putText(block, f"Name: {name}", (scaled(10), scaled(15)), cv2.FONT_HERSHEY_SIMPLEX, 0.3 * UI_SCALE, (0, 0, 0), 1)
            cv2.putText(block, f"ID: {uid}", (scaled(10), scaled(30)), cv2.FONT_HERSHEY_SIMPLEX, 0.25 * UI_SCALE, (0, 0, 0), 1)
            cv2.putText(block, f"Conf: {conf_text}", (scaled(10), scaled(45)), cv2.FONT_HERSHEY_SIMPLEX, 0.25 * UI_SCALE, (0, 0, 0), 1)

            info_column[y_start:y_end, :, :] = block
        else:
            face_column[y_start:y_end, :, :] = 255
            info_column[y_start:y_end, :, :] = 255

        # Vẽ separator
        if i < max_faces - 1:
            face_column[y_end:y_end+sep, :, :] = separator_color
            info_column[y_end:y_end+sep, :, :] = separator_color

    final_display = np.hstack((video_frame, face_column, info_column))

    # Vẽ line phân cách giữa các cột
    x1 = video_frame.shape[1]
    x2 = x1 + face_column.shape[1]
    cv2.line(final_display, (x1, 0), (x1, final_display.shape[0]), separator_color, sep)
    cv2.line(final_display, (x2, 0), (x2, final_display.shape[0]), separator_color, sep)

    # Hiển thị tọa độ chuột nếu có
    if mouse_pos is not None:
        mx, my = mouse_pos
        if 0 <= mx < video_w and 0 <= my < video_h:
            scale_x = original_w / video_w
            scale_y = original_h / video_h
            ori_x = int(mx * scale_x)
            ori_y = int(my * scale_y)
            text_pos = (mx + 10, my - 10 if my > 20 else my + 20)
            cv2.rectangle(final_display, (text_pos[0] - 5, text_pos[1] - 20),
                          (text_pos[0] + 110, text_pos[1] + 5), (255, 255, 255), -1)
            cv2.putText(final_display, f"Pos: ({ori_x}, {ori_y})", text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return final_display
