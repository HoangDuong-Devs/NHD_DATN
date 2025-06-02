import os
from trackers.object_tracker import ObjectTracker 
from config.cfg_py import config
os.environ["YOLO_VERBOSE"] = "FALSE"

from ultralytics import YOLO  

class HumanDetectionPredictor:
    def __init__(self, 
                 yolo_model_path=config.get("human_detection_model.model_path", None),
                 detect_interval=config.get("human_detection_model.detect_interval", 1), 
                 min_detection_conf=config.get("human_detection_model.min_detection_confidence", 0.25),
                 use_tracking_prediction=False
    ):
        self.yolo_model_path = yolo_model_path
        self.detect_interval = detect_interval
        self.min_detection_conf = min_detection_conf
        self.use_tracking_prediction = use_tracking_prediction

        # Khởi tạo mô hình YOLO và tracker
        self.yolo_model = YOLO(self.yolo_model_path)
        self.tracker = ObjectTracker(max_age=30, n_init=5)

        # Lưu trạng thái trước
        self.boxes_yolo_prev = []
        self.tracks_prev = {}

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def iou_bbox_deepsort_yolo(self, tracks_prev, yolo_bboxes, iou_threshold=0.3):
        matched_bboxes = []
        used_yolo_idx = set()

        for track_id, box_ds in tracks_prev.items():
            best_iou = 0
            best_idx = None
            for idx, (box_yolo, conf, cls) in enumerate(yolo_bboxes):
                if idx in used_yolo_idx:
                    continue
                x, y, w, h = box_yolo
                box_y = [x, y, x + w, y + h]
                iou = self.compute_iou(box_ds, box_y)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            if best_iou >= iou_threshold and best_idx is not None:
                matched_bboxes.append(yolo_bboxes[best_idx])
                used_yolo_idx.add(best_idx)

        unmatched_bboxes = [yolo_bboxes[i] for i in range(len(yolo_bboxes)) if i not in used_yolo_idx]

        return matched_bboxes + unmatched_bboxes

    def predict(self, frame, frame_idx):
        use_yolo = (len(self.tracks_prev) == 0) or (frame_idx % self.detect_interval == 0)

        if use_yolo:
            boxes_yolo = []
            results = self.yolo_model.predict(frame, device="cuda", conf=self.min_detection_conf, iou=0.3, verbose=False)[0]
            
            if results.boxes is not None:
                for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
                    if int(cls) == 0:
                        x1, y1, x2, y2 = map(int, box[:4].tolist())
                        w, h = x2 - x1, y2 - y1
                        boxes_yolo.append(([x1, y1, w, h], float(conf), 0))  # class 0: person

            if self.use_tracking_prediction and self.tracks_prev:
                boxes_input = self.iou_bbox_deepsort_yolo(self.tracks_prev, boxes_yolo, iou_threshold=0.15)
            else:
                boxes_input = boxes_yolo

            self.boxes_yolo_prev = boxes_input
        else:
            boxes_input = self.boxes_yolo_prev

        # Cập nhật tracker
        tracks, tracks_prev = self.tracker.update_tracks(boxes_input, frame)
        self.tracks_prev = tracks_prev

        # Chuẩn bị output
        boxes, confs, clss, ids = [], [], [], []
        for track in tracks:
            if track.is_confirmed():
                l, t, r, b = map(int, track.to_ltrb())
                boxes.append([l, t, r, b])
                confs.append(1.0)
                clss.append(0) 
                ids.append(track.track_id)

        return boxes, confs, clss, ids
