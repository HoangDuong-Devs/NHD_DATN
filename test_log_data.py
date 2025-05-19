from services.intrusion_monitor import IntrusionAlertService
import cv2
import numpy as np
import time
from schemas.area import *
from config.cfg_py import config
from database.mongodb_client import Database


def test_database():
    # Khởi tạo kết nối (nếu chưa khởi tạo)
    Database.initialize(config.get("database.mongo_uri", None), "IntrusionAlert")

    # Test insert
    record = {
        "name": "Intruder1",
        "time": "2025-05-19 12:00:00",
        "location": "Gate A",
        "alert_level": 5
    }
    inserted_id = Database.insert("alerts", record, unique_fields=["name", "time"])
    print("Insert result:", inserted_id)

    # Test insert trùng (nên trả về "Already exists")
    duplicate_result = Database.insert("alerts", record, unique_fields=["name", "time"])
    print("Insert duplicate result:", duplicate_result)

    # Test update
    filters = {"name": "Intruder1", "time": "2025-05-19 12:00:00"}
    update_fields = {"alert_level": 7, "resolved": False}
    update_result = Database.update("alerts", filters, record=update_fields, upsert=False)
    print("Update result:", update_result)

if __name__ == "__main__":
    test_database()
