from pymongo import MongoClient
import os
DB_HOST = os.getenv("DB_HOST")

def get_data():

    client = MongoClient(f'{DB_HOST}')
    db = client['mlops']
    collection = db['master']
    cursor = collection.find({})
    return cursor;