from pymongo import MongoClient

def save_to_mongodb(data):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['mlops']
    collection = db['master']
    collection.insert_one(data)
