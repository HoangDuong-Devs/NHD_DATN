import cv2 
import numpy as np
        
def intersect_polygon_test(polygon, box):
    """
    Kiểm tra xem một bounding box có giao với polygon hay không.

    Args:
        polygon (list of tuple): Danh sách các điểm (x, y) tạo thành đa giác (polygon).
        box (numpy.ndarray hoặc tuple): Bounding box dưới dạng (x_min, y_min, x_max, y_max).

    Returns:
        bool: 
            - True nếu bounding box và polygon có phần giao nhau (ít nhất một điểm của polygon nằm trong bounding box
              hoặc ít nhất một điểm của bounding box nằm trong polygon).
            - False nếu không có giao nhau.
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

    
    