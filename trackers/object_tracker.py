from deep_sort_realtime.deepsort_tracker import DeepSort
from config.cfg_py import config

class ObjectTracker:
    def __init__(self, max_age=30, n_init=2, max_cosine_distance=0.4, nn_budget=100):
        self.tracker = DeepSort(max_age=max_age, n_init=n_init, max_cosine_distance=max_cosine_distance, nn_budget=nn_budget)
        self.tracks_prev = {}

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def match_deepsort_with_yolo(self, track_bboxes, yolo_bboxes, iou_thresh=0.1):
        matched = []
        used_yolo = set()

        for track_id, box_ds in track_bboxes.items():
            best_iou, best_yolo = 0, None
            for idx, (box_yolo, _, _) in enumerate(yolo_bboxes):
                if idx in used_yolo:
                    continue
                x1, y1, w, h = box_yolo
                box_y = [x1, y1, x1 + w, y1 + h]
                iou = self.compute_iou(box_ds, box_y)
                if iou > best_iou:
                    best_iou = iou
                    best_yolo = idx

            if best_iou >= iou_thresh and best_yolo is not None:
                matched.append(yolo_bboxes[best_yolo])  # giữ box YOLO
                used_yolo.add(best_yolo)

        # Thêm box chưa match (coi là object mới)
        unmatched = [yolo_bboxes[i] for i in range(len(yolo_bboxes)) if i not in used_yolo]
        return matched + unmatched

    def update_tracks(self, boxes_input, frame):
        # Update tracks with new boxes
        tracks = self.tracker.update_tracks(boxes_input, frame=frame)

        # Lưu bbox track lại để so sánh lần detect sau
        tracks_prev = {}
        for track in tracks:
            if track.is_confirmed():
                l, t, r, b = map(int, track.to_ltrb())
                tracks_prev[track.track_id] = [l, t, r, b]

        return tracks, tracks_prev
