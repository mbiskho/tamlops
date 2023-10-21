from pymongo import MongoClient

def get_data():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['mlops']
    collection = db['master']
    cursor = collection.find({})
    return cursor;