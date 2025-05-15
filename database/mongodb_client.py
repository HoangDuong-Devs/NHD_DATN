from pymongo import MongoClient

class Database:
    _client = None
    _db = None

    @staticmethod
    def initialize(uri="mongodb://localhost:27017", db_name="mydatabase"):
        if Database._client is None:
            Database._client = MongoClient(uri)
            Database._db = Database._client[db_name]
            print("MongoDB connected.")

    @staticmethod
    def insert(table, record, unique_fields=None):
        coll = Database._db[table]
        # Nếu có unique_fields, bạn có thể check document tồn tại rồi update hoặc skip
        if unique_fields:
            query = {field: record.get(field) for field in unique_fields}
            existing = coll.find_one(query)
            if existing:
                return "Already exists"
        result = coll.insert_one(record)
        return result.inserted_id

    @staticmethod
    def update(table, filters, record=None, upsert=True):
        coll = Database._db[table]
        # record là dict chứa fields cần update
        update_doc = {"$set": record} if record else {}
        result = coll.update_one(filters, update_doc, upsert=upsert)
        return f"Matched: {result.matched_count}, Modified: {result.modified_count}"
