import numpy as np
from database.minio_client import MinioClient

def log_black_square(minio_client: MinioClient, bucket_name: str, destination_file: str):
    # Tạo ảnh đen 224x224 (3 kênh màu)
    black_img = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Upload ảnh lên MinIO
    success = minio_client.upload_file(black_img, bucket_name, destination_file, content_type="image/jpeg")
    
    if success:
        print(f"Logged black square image to {bucket_name}/{destination_file}")
    else:
        print("Failed to log black square image.")

# Ví dụ sử dụng
if __name__ == "__main__":
    endpoint = "your-minio-endpoint:9000"
    access_key = "your-access-key"
    secret_key = "your-secret-key"
    bucket = "test-bucket"
    file_name = "black_square.jpg"

    client = MinioClient(endpoint, access_key, secret_key, secure=False)
    log_black_square(client, bucket, file_name)
