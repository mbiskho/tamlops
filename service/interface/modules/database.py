from pymongo import MongoClient
import os

DB = os.getenv("DB")

def save_to_mongodb(data):
    client = MongoClient(f'{DB}')
    db = client['mlops']
    collection = db['master']
    collection.insert_one(data)
