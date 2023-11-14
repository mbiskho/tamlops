import asyncpg
from dotenv import load_dotenv
import os
import json

load_dotenv()

DB = os.environ.get("DB")

async def save_training_db(type, file, size, params):
    conn = await asyncpg.connect(DB)
    await conn.execute("INSERT INTO training_queue (type, file, size, params) VALUES ($1, $2, $3, $4)", type, file, size, params)
    await conn.close()
    return {"message": "Training dataset saved successfully"}
