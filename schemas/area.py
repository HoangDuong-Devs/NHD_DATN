import numpy as np
import cv2
class Area:
    def __init__(self, contour, area_id):
        self.contour = np.array(contour, dtype=np.int32)  # Tọa độ các điểm của polygon
        self.count   = 0                                  # Số lượng đối tượng xuất hiện trong khu vực
        self.area_id = area_id                            # Id khu vực

    def __repr__(self):
        return f"Area(area_id={self.area_id}, contour={self.contour})"
    

def draw_areas(img, areas, intrusion_count: int = 0):
    """
    Vẽ các khu vực lên ảnh hoặc tô đỏ ảnh nếu không có khu vực nhưng có đối tượng xâm nhập.

    Parameters:
        img (np.ndarray): Ảnh đầu vào.
        areas (list | None): Danh sách các khu vực hoặc None.
        intrusion_count (int): Số lượng đối tượng xâm nhập (mặc định 0).
    """
    if not areas:
        text = f"Intrusions: {intrusion_count}"
        cv2.putText(img, text, (img.shape[1] - 220, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2)

        if intrusion_count > 0:
            red_overlay = np.full_like(img, (0, 0, 150))
            img[:] = cv2.addWeighted(img, 0.7, red_overlay, 0.3, 0)
        return

    for area in areas:
        id_int = int(area.area_id)
        color = (
            (((~id_int) << 6) & 0x100)  - 1,
            (((~id_int) << 7) & 0x0100) - 1,
            (((~id_int) << 8) & 0x0100) - 1,
        )
        if area.count > 0:
            color = (255 - color[0], 255 - color[1], 255 - color[2])
        cv2.polylines(img, [area.contour], True, color, 2)

        text = f"Area {area.area_id}: {area.count}"
        cv2.putText(
            img,
            text,
            (area.contour[0][0], area.contour[0][1]),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            color,
            2,
        )
