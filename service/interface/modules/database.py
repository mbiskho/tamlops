import asyncpg
from dotenv import load_dotenv
import os

load_dotenv()

DB = os.environ.get("DB")

async def save_training_db(type, file):
    conn = await asyncpg.connect(DB)
    await conn.execute("INSERT INTO training_queue (type, file) VALUES ($1, $2)", type, file)
    await conn.close()
    return {"message": "Training dataset saved successfully"}
