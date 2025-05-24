import os
from trackers.object_tracker import ObjectTracker 
from config.cfg_py import config
os.environ["YOLO_VERBOSE"] = "FALSE"

from ultralytics import YOLOE

class HumanDetectionPredictor:
    def __init__(self, 
                 yolo_model_path="yoloe-11l-seg.pt", 
                 detect_interval=3, 
                 use_tracking_prediction=False
    ):
        self.yolo_model_path = yolo_model_path
        self.detect_interval = detect_interval
        self.use_tracking_prediction = use_tracking_prediction

        # Khởi tạo mô hình YOLO và tracker
        self.yolo_model = YOLOE(self.yolo_model_path)
        self.yolo_model.set_classes(["person"], self.yolo_model.get_text_pe(["person"]))
        self.tracker = ObjectTracker(max_age=15, n_init=2)

        # Nếu cấu hình dùng dự đoán thì mới cần lưu
        self.boxes_yolo_prev = []
        self.tracks_prev = []

    def predict(self, frame, frame_idx):
        use_yolo = (len(self.tracks_prev) == 0) or (frame_idx % self.detect_interval == 0)

        if use_yolo:
            # === YOLO Detection ===
            boxes_yolo = []
            results = self.yolo_model.predict(frame, device="cuda", conf=0.5, iou=0.5, verbose=False)[0]
            if results.boxes is not None:
                for box, conf in zip(results.boxes.xyxy, results.boxes.conf):
                    x1, y1, x2, y2 = map(int, box[:4].tolist())
                    boxes_yolo.append(([x1, y1, x2 - x1, y2 - y1], float(conf), 0))  # cls=0 là 'person'

            if self.use_tracking_prediction and self.tracks_prev:
                boxes_input = self.tracker.match_deepsort_with_yolo(self.tracks_prev, boxes_yolo)
            else:
                boxes_input = boxes_yolo

            self.boxes_yolo_prev = boxes_input
        else:
            boxes_input = self.boxes_yolo_prev

        # Cập nhật track
        tracks, tracks_prev = self.tracker.update_tracks(boxes_input, frame)

        if self.use_tracking_prediction:
            self.tracks_prev = tracks_prev
        else:
            self.tracks_prev = []  # Không lưu lại để tránh dự đoán tiếp

        # Chuẩn bị output: boxes, confs, clss, ids
        boxes, confs, clss, ids = [], [], [], []

        for track in tracks:
            if track.is_confirmed():
                l, t, r, b = map(int, track.to_ltrb())
                boxes.append([l, t, r, b])
                confs.append(1.0)  # Có thể dùng conf từ YOLO nếu cần
                clss.append(0)  # class = 0, do chỉ có "person"
                ids.append(track.track_id)

        return boxes, confs, clss, ids

