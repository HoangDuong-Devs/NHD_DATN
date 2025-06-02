from database.mongodb_client import Database
from database.minio_client import *
import uuid
from datetime import datetime 
from config.cfg_py import config
import cv2
class LogResults():
    def __init__(self):
        self.init_database = False
        self.table_name    = config.get("database.table_name"  , "alerts")
        self.minio_client  = None
        
        self.minio_client = MinioClient()
        self._init_database_()
        
        self._previous_intrusion_states = {}
        self._previous_time_stamp       = None
        
    def _init_database_(self):
        Database.initialize(
            config.get("database.mongo_uri", "mongodb://localhost:27017"), 
            config.get("database.db_name"  , "intrusion_db")
        )
        current_time = datetime.now()
        current_date = current_time.strftime("%Y-%m-%d")
        record = {
            "date": current_date
        }
        self.filters = record
        if not self.init_database:
            try:
                res = Database.insert(table=self.table_name, record=record, unique_fields=["date"])
                if res == "Already exists":
                    print("Database already initialized.")
                self.init_database = True
            except Exception as e:
                print(f"Error initializing database: {e}")
                return
            
    def _intrusion_alert_log(self, res):
        if res is not None:
            timestamp = res["timestamp"].split(".")[0]
            result = res['intrusion_results']
            
            try:
                temp_result = {zone_id: zone_info.get('intrusion', False) for zone_id, zone_info in result.items()}
                
                # Chuẩn bị các thông tin thời gian
                dt_object = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                date_str = dt_object.strftime("%Y-%m-%d")
                hour_str = dt_object.strftime("%H")
                time_str = dt_object.strftime("%H:%M:%S").replace(":", "-")
                
                for zone_id, zone_information in result.items():
                    images = zone_information.get('cropped_images', {}) 
                    for id_obj, obj_information in images.items():
                        try:
                            image = obj_information["image"]
                            name = obj_information.get("name", "unknown")
                            file_name = f"{time_str}_{name}_{int(id_obj)}.jpg"
                            minio_des = f"{date_str}/{hour_str}/{zone_id}/{file_name}"
                            
                            self.minio_client.upload_file(
                                data            = cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                                bucket_name     = config.get("minio.bucket_name", "intrusion-images"),
                                destination_file=minio_des,
                            )
                        except Exception as img_error:
                            print(f"Failed to upload image for id={id_obj}, zone_id={zone_id}: {img_error}")
                            
                if temp_result != self._previous_intrusion_states and time_str != self._previous_time_stamp:
                    successfully_logged = False
                    try:
                        for zone_id, zone_information in temp_result.items():
                            if zone_information and not self._previous_intrusion_states.get(zone_id, False):
                                unique_id = f"zone-{zone_id}-{uuid.uuid4()}"
                                record = {
                                    "date"     : date_str,
                                    "zone_id"  : zone_id,
                                    "unique_id": unique_id,
                                    "timepoint": time_str
                                }
                                update_doc = {"$push": {"intrusion_logs": record}}
                                # Ghi vào MongoDB
                                Database._db[self.table_name].update_one(self.filters, update_doc, upsert=True)
                        successfully_logged = True
                    except Exception as e:
                        print(f"Error logging to MongoDB: {e}")
                    if successfully_logged:
                        self._previous_time_stamp = time_str
                        self._previous_intrusion_states = {**self._previous_intrusion_states, **temp_result}
                        
            except Exception as e:
                print(f"Error processing intrusion results: {e}")
                return