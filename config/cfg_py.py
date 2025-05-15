import yaml

class Config:
    def __init__(self, config_file="config/cfg.yaml"):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_file, 'r') as f:
            return yaml.safe_load(f)

    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default)
        return value

# Khởi tạo cấu hình
config = Config()

# Lấy thông tin từ cấu hình cho face detection model
face_detection_name = config.get("face_detection_model.name", "mediapipe")
model_selection     = config.get("face_detection_model.model_selection", 1)
detection_conf      = config.get("face_detection_model.min_detection_confidence", 0.5)
refine_landmarks    = config.get("face_detection_model.refine_landmarks", True)

# Lấy thông tin từ cấu hình cho database
mongo_uri = config.get("database.mongo_uri", "mongodb://localhost:27017")
db_name   = config.get("database.db_name"  , "intrusion_db")

# Lấy thông tin từ cấu hình cho MinIO
minio_endpoint    = config.get("minio.endpoint"   , "minio:9000")
minio_access_key  = config.get("minio.access_key" , "minioadmin")
minio_secret_key  = config.get("minio.secret_key" , "minioadmin")
minio_bucket_name = config.get("minio.bucket_name", "intrusion-images")

# Lấy thông tin cấu hình tracker
tracker_max_age = config.get("tracker.max_age", 30)
tracker_n_init  = config.get("tracker.n_init" , 3)

# In ra để kiểm tra
print(f"Face Detection Model    : {face_detection_name}")
print(f"Model Selection         : {model_selection}")
print(f"Min Detection Confidence: {detection_conf}")
print(f"Refine Landmarks        : {refine_landmarks}")
print(f"Mongo URI               : {mongo_uri}")
print(f"MinIO Endpoint          : {minio_endpoint}")
print(f"tracker max_age         : {tracker_max_age}")
print(f"tracker n_init          : {tracker_n_init}")
