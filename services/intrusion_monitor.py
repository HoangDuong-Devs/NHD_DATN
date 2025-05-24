import os
import numpy as np
from datetime import datetime, timedelta
from detectors.object_detector import HumanDetectionPredictor
from detectors.face_detector import FaceDetectionPredictor
from face_recognition_module.recognition import FaceRecognitionModule
from utils.face_alignment import *
import cv2
from schemas.area import *
from schemas.object import *
from schemas.intrusion_event import * 

class IntrusionAlertService:
    def __init__(self, detector=None, **kwargs):
        self.objects                  = {}                       # Từ điển đối tượng đã xuất hiện 
        self.differences              = {}                       # Từ điển lượng thời gian biến mất của các đối tượng
        self.cropped_ids              = set()                    # set đối tượng đã được capture
        self.areas                    = kwargs.get("areas", [])  # Danh sách các khu vực   
        self.sub_detector             = None                                 
        self.setup()
        self.initialize_detector(detector)

    def setup(self):
        self.frame_appearance = 50   # Số frame xuất hiện của một đối tượng để thực hiện capture
        self.time_out         = 4    # Thời gian giới hạn đối tượng biến mất
    def initialize_detector(self, detector):
        self.human_detector   = HumanDetectionPredictor(yolo_model_path="yoloe-11l-seg.pt", detect_interval=1)          
        self.face_detector    = FaceDetectionPredictor()
        # self.face_recognition = FaceRecognitionModule()
        self.aligner          = FaceAligner(output_size=150)

            
    def model_inference(self, image, **kwargs):
        return self.human_detector.predict(image, **kwargs)
    
    def sub_model_inference(self, image, **kwargs):
        return self.face_detector.predict(image, **kwargs)
    
    # Cài đặt dịch vụ
    def service_implement(self, frame, idx):
        boxes, confs, clss, ids = self.model_inference(frame, frame_idx=idx)
        print(boxes)
        results = extract_aligned_faces_from_people(
            frame=frame,
            boxes=boxes,
            ids=ids,
            face_detector=self.face_detector,       # FaceDetectionPredictor của bạn
            face_aligner=self.aligner  # Aligner bạn đang dùng
        )
        image_origin = frame.copy()
        visualize_boxes_with_ids(frame, boxes, ids, color=(0, 255, 0), thickness=2)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        temp_objects    = {}
        
        intrude_results = {}
        if not self.areas:
            captured_images = {}  # Biến lưu giữ hình ảnh tạm cho Non-Area
        
            for i, (box, id) in enumerate(zip(boxes, ids)):
                update_objects(self.objects, id, box, image_origin)
                
                # Lưu trữ các đối tượng xuất hiện tại thời điểm hiện tại
                temp_objects[id] = self.objects[id]
                print(self.objects[id].appearance_count)
                # Kiểm tra điều kiện để capture đối tượng
                if (
                    (self.objects[id].appearance_count) >= self.frame_appearance
                    and id not in self.cropped_ids
                    and self.objects[id].latest_image is not None
                ):
                    captured_images[id] = get_captured_image(self.objects, id)
                    self.cropped_ids.add(id)
        
            # Tối ưu xử lý đối tượng biến mất đột ngột (Xuất hiện ko đủ số frames yêu cầu)
            for i in set(self.objects.keys()) - set(temp_objects.keys()) - self.cropped_ids:
                self.differences[i] = datetime.now() - self.objects[i].latest_time
                if self.differences[i] > timedelta(seconds=self.time_out) and self.objects[i].latest_image is not None:
                    captured_images[i] = get_captured_image(self.objects, i)
                    self.cropped_ids.add(i)
                    
            # Kiểm tra trạng thái xâm nhập
            all_familiar = all(obj.is_familiar for obj in temp_objects.values())
            intrude_results["0"] = {
                "intrusion": not all_familiar,
                "cropped_images": captured_images
            }
        else:
            intrude_results = {}
            draw_areas(frame, self.areas)
            for area in self.areas:
                area.count = 0
                temp_objects_in_area = {}  # Lưu các đối tượng xuất hiện trong khu vực hiện tại
                captured_images = {}       # Lưu ảnh đã cắt cho khu vực này
                
                # Giải quyết trường hợp có Khu vực (Area) và Nhận dạng khuôn mặt (Face recognition)
                if self.face_recognition_feature:                        
                    for i, (box, id) in enumerate(zip(boxes, ids)):
                        update_objects(self.objects, id, box, image_origin)
                        
                        if not self.objects[id].is_familiar:
                            pass                                
                                
                        if intersect_polygon_test(area.contour, box):
                            temp_objects_in_area[id] = self.objects[id]
                            area.count += 1
                            self.objects[id].update_intrude(area.area_id, timestamp)

                            if (
                                (self.objects[id].appearance_count >= self.frame_appearance or
                                self.objects[id].is_familiar)
                                and not (id, area.area_id) in self.cropped_ids
                                and self.objects[id].latest_image is not None
                            ):
                                captured_images[id] = get_captured_image(self.objects, id)
                                self.cropped_ids.add((id, area.area_id))
                
                for i in set(self.objects.keys()) - set(temp_objects_in_area.keys()):
                    if (i, area.area_id) not in self.cropped_ids and self.objects[i].intrude.get(area.area_id, False) and self.objects[i].latest_image is not None:
                        self.differences[i] = datetime.now() - self.objects[i].latest_time
                        if self.differences[i] > timedelta(seconds=self.time_out):
                            captured_images[i] = get_captured_image(self.objects, i)
                            self.cropped_ids.add((i, area.area_id))

                # Kiểm tra kết quả xâm nhập cho khu vực hiện tại
                intrusion_status = area.count > 0 and not all(obj.is_familiar for obj in temp_objects_in_area.values())
                intrude_results[area.area_id] = {
                    "intrusion": intrusion_status,
                    "cropped_images": captured_images
                }
        
        format_result = {
            "image": frame,
            "timestamp": timestamp,
            "intrusion_results": intrude_results
        }
        
        return format_result
        