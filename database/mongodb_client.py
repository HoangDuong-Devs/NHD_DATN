from pymongo import MongoClient
from config.cfg_py import config

class Database:
    _client = None
    _db = None

    @staticmethod
    def initialize(uri=config.get("database.mongo_uri", "mongodb://localhost:27017"), 
                   db_name=config.get("database.db_name", "intrusion_db")):
        """
        Khởi tạo kết nối MongoDB với URI và tên database được cung cấp hoặc mặc định.
        
        Args:
            uri (str): Chuỗi kết nối MongoDB.
            db_name (str): Tên database cần kết nối.
        """
        if Database._client is None:
            Database._client = MongoClient(uri)
            Database._db = Database._client[db_name]
            print("MongoDB connected.")

    @staticmethod
    def insert(table, record, unique_fields=None):
        """
        Chèn bản ghi mới vào collection. Nếu chỉ định unique_fields, kiểm tra tồn tại trước khi chèn.
        
        Args:
            table (str): Tên collection.
            record (dict): Bản ghi dữ liệu cần chèn.
            unique_fields (list|None): Danh sách trường để kiểm tra duy nhất.
        
        Returns:
            inserted_id nếu chèn thành công hoặc chuỗi "Already exists" nếu đã tồn tại.
        """
        coll = Database._db[table]
        if unique_fields:
            query = {field: record.get(field) for field in unique_fields}
            existing = coll.find_one(query)
            if existing:
                return "Already exists"
        result = coll.insert_one(record)
        return result.inserted_id

    @staticmethod
    def update(table, filters, record=None, upsert=True):
        """
        Cập nhật một bản ghi trong collection dựa trên điều kiện lọc, hoặc chèn mới nếu không tìm thấy (upsert).
        
        Args:
            table (str): Tên collection.
            filters (dict): Điều kiện lọc để tìm bản ghi cần cập nhật.
            record (dict|None): Dữ liệu cập nhật (giá trị của các trường).
            upsert (bool): Nếu True thì thêm mới nếu không tồn tại.
        
        Returns:
            str: Thông tin số lượng bản ghi được match và sửa đổi.
        """
        coll = Database._db[table]
        update_doc = {"$set": record} if record else {}
        result = coll.update_one(filters, update_doc, upsert=upsert)
        return f"Matched: {result.matched_count}, Modified: {result.modified_count}"
