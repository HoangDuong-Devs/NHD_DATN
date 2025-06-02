import numpy as np
from PIL import Image
import io
import os
from minio import Minio
from minio.error import S3Error
from config.cfg_py import *

class MinioClient:
    def __init__(
        self, 
        endpoint  : str = config.get("minio.endpoint"   , "minio:9000"), 
        access_key: str = config.get("minio.access_key" , "minioadmin"), 
        secret_key: str = config.get("minio.secret_key" , "minioadmin"), 
        secure    : bool= False
    ):
        self.minio = Minio(endpoint  =endpoint, 
                           access_key=access_key, 
                           secret_key=secret_key, 
                           secure    =secure)

    def upload_file(self, data, bucket_name, destination_file, content_type=None):
        """_Tải dữ liệu lên MinIO bucket với hỗ trợ nhiều định dạng dữ liệu_

        Args:
            data (_np.ndarray | bytes | file-like object_): _Dữ liệu cần tải lên (ảnh NumPy, bytes, hoặc file-like object)_
            bucket_name (_str_): _Tên MinIO bucket để lưu trữ dữ liệu_
            destination_file (_str_): _Tên tệp đích trong bucket_
            content_type (_str_, optional): _Loại nội dung MIME (ví dụ: 'image/jpeg', 'application/octet-stream'). Tự động xác định nếu không cung cấp._

        Returns:
            _bool_: _True nếu tải lên thành công, False nếu thất bại_
        """
        try:
            if not self.minio.bucket_exists(bucket_name):
                self.minio.make_bucket(bucket_name)
            
            if isinstance(data, np.ndarray):
                if content_type is None:
                    content_type = "image/jpeg"
                image = Image.fromarray(data)
                image_bytes = io.BytesIO()
                image.save(image_bytes, format="JPEG")
                image_bytes.seek(0)
                data = image_bytes
                length = image_bytes.getbuffer().nbytes
                
            elif isinstance(data, bytes):
                if content_type is None:
                    content_type = "application/octet-stream"
                data = io.BytesIO(data)
                length = len(data.getvalue())
            
            elif hasattr(data, "read"):
                if content_type is None:
                    content_type = "application/octet-stream"
                length = os.fstat(data.fileno()).st_size
            
            else:
                raise ValueError("Unsupported data type for upload_file.")
            
            self.minio.put_object(bucket_name, destination_file, data, length, content_type)
            print(f"Successfully uploaded '{destination_file}' to bucket '{bucket_name}'.")
            return True
        except S3Error as err:
            print(f"Failed to upload: {err}")
            return False
        except Exception as ex:
            print(f"Failed to upload: {ex}")
            return False

    def crawl_data(self, bucket_name, prefix=None, local_dir="./downloaded"):
        """_Tải toàn bộ đối tượng từ MinIO bucket về thư mục cục bộ_

        Args:
            bucket_name (_str_): _Tên của MinIO bucket cần tải dữ liệu_
            prefix (_str_, optional): _Tiền tố để lọc các object cần tải (thư mục con). Mặc định là None._
            local_dir (_str_, optional): _Thư mục cục bộ để lưu dữ liệu tải về. Mặc định là "./downloaded"._

        Returns:
            None
        """
        if prefix:
            objects = self.minio.list_objects(bucket_name, prefix=prefix, recursive=True)
        else:
            objects = self.minio.list_objects(bucket_name, recursive=True)

        os.makedirs(local_dir, exist_ok=True)
        count = 0
        for obj in objects:
            local_path = os.path.join(local_dir, os.path.basename(obj.object_name))
            self.minio.fget_object(bucket_name, obj.object_name, local_path)
            print(f"Downloaded {obj.object_name} to {local_path}")
            count += 1
        print(f"Downloaded {count} objects from bucket '{bucket_name}'.")