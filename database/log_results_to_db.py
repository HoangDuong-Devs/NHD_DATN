from database.mongodb_client import Database
from database.minio_client import *
import uuid
import base64
from datetime import datetime 
from config.cfg_py import config

class LogResults():
    def __init__(self):
        self.init_database = False
        self.table_name = None
        self.minio_client = None
        self._previous_intrusion_states = {}
        self._previous_time_stamp = None
        self._init_database_()
        
    def _init_database_(self):
        Database.initialize(config.get("database.mongo_uri", None), "IntrusionAlert")
        current_time = datetime.now()
        current_date = current_time.strftime("%Y-%m-%d")
        record = {
            "date": current_date
        }
        self.filters = record
        if not self.init_database:
            try:
                res = Database.insert(table="alerts", record=record, unique_fields=["date"])
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
                dt_object = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                date_str = dt_object.strftime("%Y-%m-%d")
                time_str = dt_object.strftime("%H:%M:%S").replace(":", "-")
                
                if temp_result != self._previous_intrusion_states and time_str != self._previous_time_stamp:
                    successfully_logged = False
                    try:
                        for zone_id, zone_information in temp_result.items():
                            if zone_information and not self._previous_intrusion_states.get(zone_id, False):
                                unique_id = f"zone-{zone_id}-{uuid.uuid4()}"
                                record = {
                                    "date": date_str,
                                    "zone_id": zone_id,
                                    "unique_id": unique_id,
                                    "timepoint": time_str
                                }
                                update_doc = {"$push": {"intrusion_logs": record}}
                                # Ghi vào MongoDB (update với filters)
                                Database._db["alerts"].update_one(self.filters, update_doc, upsert=True)
                        successfully_logged = True
                    except Exception as e:
                        print(f"Error logging to MongoDB: {e}")
                    if successfully_logged:
                        self._previous_time_stamp = time_str
                        self._previous_intrusion_states = {**self._previous_intrusion_states, **temp_result}
                        
            except Exception as e:
                print(f"Error processing intrusion results: {e}")
                return

            