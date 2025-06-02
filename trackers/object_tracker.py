from deep_sort_realtime.deepsort_tracker import DeepSort
from config.cfg_py import config

class ObjectTracker:
    def __init__(self, max_age=30, n_init=2, max_cosine_distance=0.5, nn_budget=100):
        self.tracker = DeepSort(max_age=max_age, n_init=n_init, max_cosine_distance=max_cosine_distance, nn_budget=nn_budget)
        self.tracks_prev = {}

    def compute_iou(self, boxA, boxB):
        """
        Tính Intersection over Union (IoU) giữa hai bounding box.

        Args:
            boxA (tuple hoặc list): Bounding box thứ nhất dưới dạng (x_min, y_min, x_max, y_max).
            boxB (tuple hoặc list): Bounding box thứ hai dưới dạng (x_min, y_min, x_max, y_max).

        Returns:
            float: Giá trị IoU nằm trong khoảng [0, 1], thể hiện tỷ lệ giao giữa hai bounding box so với phần hợp nhất của chúng.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def update_tracks(self, boxes_input, frame):
        """
        Cập nhật tracker với các bounding box mới, trả về danh sách các track hiện tại và dictionary chứa bounding box của từng track.

        Args:
            boxes_input (list): Danh sách các bounding box đầu vào để cập nhật tracker (thường là kết quả từ detector).
            frame (numpy.ndarray): Khung hình hiện tại (dùng để giới hạn bbox trong kích thước ảnh).

        Returns:
            tracks (list): Danh sách các đối tượng track sau khi cập nhật, mỗi track có trạng thái và bbox.
            tracks_prev (dict): Dictionary dạng {track_id: [left, top, right, bottom]} chứa bounding box đã clamp (giới hạn trong ảnh) của từng track đã được xác nhận (confirmed).
        """
        # Update tracks with new boxes
        tracks = self.tracker.update_tracks(boxes_input, frame=frame)

        # Lưu bbox track lại để so sánh lần detect sau
        h, w = frame.shape[:2]
        tracks_prev = {}

        for track in tracks:
            if track.is_confirmed():
                l, t, r, b = map(int, track.to_ltrb())

                l = max(0, min(l, w))
                r = max(0, min(r, w))
                t = max(0, min(t, h))
                b = max(0, min(b, h))

                # Bỏ qua bbox không hợp lệ
                if r <= l or b <= t:
                    continue

                tracks_prev[track.track_id] = [l, t, r, b]

        return tracks, tracks_prev

