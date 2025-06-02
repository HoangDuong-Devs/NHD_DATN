import cv2
import time
from schemas.area import *
from utils.visualize import *
from database.log_results_to_db import LogResults
from services.intrusion_monitor import IntrusionAlertService

video_path = r"F:\VScode_NHD\intrusion_alert\test.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# ======= Khu vực cảnh báo mẫu =======
areas = [
    # Area(contour=[(300, 100), (600, 100), (600, 500), (300, 500)], area_id="1"),
    # Area(contour=[(400, 100), (600, 100), (600, 300), (400, 300)], area_id="2")
]

mouse_pos = [None]

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos[0] = (x, y)

def main():

    log_res = LogResults()
    service = IntrusionAlertService(areas=areas)

    out_ui = cv2.VideoWriter(
        "output_video_ui.avi",
        cv2.VideoWriter_fourcc(*'XVID'),
        fps,
        (width, height)
    )

    cv2.namedWindow("Inference")
    cv2.setMouseCallback("Inference", on_mouse) 

    frame_idx = 0

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Video ended or error reading frame.")
            break

        res = service.service_implement(frame, frame_idx)
        log_res._intrusion_alert_log(res)

        final_display = visualize_intrusion_interface(res, max_faces=5, mouse_pos=mouse_pos[0])

        cv2.imshow("Inference", final_display)

        if final_display.shape[1] != width or final_display.shape[0] != height:
            final_display = cv2.resize(final_display, (width, height))

        out_ui.write(final_display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC để thoát
            break

        frame_idx += 1

    cap.release()
    out_ui.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
