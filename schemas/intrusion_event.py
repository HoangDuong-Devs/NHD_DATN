import cv2 
import numpy as np
from schemas.area import Area

        
def intersect_polygon_test(polygon, box):
    """_summary_
        
    Args:
        polygon (_list_)         : _Tọa độ các đỉnh của Polygon_
        box     (_numpy.ndarray_): _Bounding box của đối tượng_

    Returns:
        _bool_: _Kiểm tra xem box đó có giao với Polygon không_
    """
    x_min, y_min, x_max, y_max = box

    for x, y in polygon:
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return True

    box_points = [(x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)]
    for x, y in box_points:
        if cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), (x, y), False) >= 0:
            return True

    return False

def crop_body(image, box, stranger_required):
    image_with_box = image.copy()
    
    xmin, ymin, xmax, ymax = map(int, box)
    height, width = image.shape[:2]  # Lấy kích thước ảnh
    
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(width, xmax)
    ymax = min(height, ymax)
    
    cropped_image = image_with_box[ymin:ymax, xmin:xmax]
    
    return cropped_image, box
    
    