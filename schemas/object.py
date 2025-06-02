from datetime import datetime
import cv2 

class Object:
    def __init__(self, id_object, box):
        self.id               = id_object       # Id của đối tượng (tracking)
        self.box              = box             # Bounding box của đối tương
        self.appearance_count = 0               # Số frame xuất hiện đối tượng
        self.latest_time      = datetime.now()  # Thời gian cuối cùng xuất hiện
        self.latest_image     = []              # Ảnh lần cuối xuất hiện
        self.is_familiar      = False           # Có quen thuộc không
        self.intrude          = dict()          # Xâm nhập những khu vực ?
        self.latest_face       = None
        self.name              = "Unknown"
        self.recognition_score = -1.0
 
    def update_box(self, box, image):
        self.appearance_count  = self.appearance_count + 1
        self.latest_image      = image
        self.latest_time       = datetime.now()
        self.box               = box

        
    def count_appearance(self):
        self.appearance_count += 1

    def set_latest_time(self):
        self.latest_time = datetime.now()
        
    def set_latest_image(self, image):
        self.latest_image = image
        
    def set_familiar(self):
        self.is_familiar = True
    
    def update_intrude(self, zone_id, timestamp):
        self.intrude[zone_id] =  timestamp
    
    
def crop_body(image, box):
    """_Cắt phần thân đối tượng từ ảnh dựa trên bounding box_

    Args:
        image (_ndarray_): _Ảnh gốc đầu vào_
        box (_list_): _Bounding box dạng [xmin, ymin, xmax, ymax]_

    Returns:
        tuple: 
            - _ndarray_: _Ảnh đã được cắt từ bounding box_
            - _list_: _Bounding box đã được giới hạn trong kích thước ảnh_
    """
    image_with_box = image.copy()
    
    height, width = image.shape[:2]
    
    xmin, ymin, xmax, ymax = map(int, box)
    
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(width, xmax)
    ymax = min(height, ymax)
    
    cropped_image = image_with_box[ymin:ymax, xmin:xmax]
    return cropped_image, box
    
def update_objects(objects, id, box, original_image):
    """_Cập nhật hoặc thêm mới một đối tượng vào danh sách theo ID_

    Args:
        objects (_dict_): _Danh sách các đối tượng đang theo dõi, dạng {id: Object}_
        id (_int_): _ID của đối tượng cần cập nhật_
        box (_list_): _Bounding box của đối tượng dạng [x1, y1, x2, y2]_
        original_image (_ndarray_): _Ảnh gốc dùng để crop phần thân đối tượng_

    Returns:
        None
    """
    if id not in objects:
        objects[id] = Object(id, box)
    cropped_image, box = crop_body(original_image, box)
    objects[id].update_box(box, cropped_image)

def visualize_boxes_with_ids(frame, boxes, ids, color=(0, 255, 0), thickness=2):
    """_Hiển thị các bounding box cùng với ID tương ứng trên khung hình_

    Args:
        frame (_ndarray_): _Khung hình đầu vào (ảnh RGB hoặc BGR)_
        boxes (_list_): _Danh sách các bounding box theo định dạng [x1, y1, x2, y2]_
        ids (_list_): _Danh sách ID tương ứng với mỗi bounding box_
        color (_tuple_, optional): _Màu sắc của bounding box và ID. Mặc định là xanh lá (0, 255, 0)_
        thickness (_int_, optional): _Độ dày của đường viền. Mặc định là 2_

    Returns:
        None
    """
    for box, obj_id in zip(boxes, ids):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f'ID: {obj_id}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        

def get_captured_image(objects, id):
    """_lấy ảnh và trạng thái nhận dạng của đối tượng_

    Args:
        objects (_dict_): _Danh sách các đối tượng đã từng xuất hiện_
        id (_type_): _Id của đối tượng hiện tại_

    Returns:
        _dict_: _Trả về ảnh và trạng thái nhận diện mới nhất của đối tượng_
    """
    return {
        "image": objects[id].latest_image,
        "name": objects[id].name,
        "recognition": objects[id].is_familiar
    }