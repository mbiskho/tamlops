import asyncpg
from dotenv import load_dotenv
import os
import json

load_dotenv()

DB = os.environ.get("DB")

async def save_training_db(type, file, size, params):
    print(DB)
    conn = await asyncpg.connect(DB)
    await conn.execute("INSERT INTO training_queue (type, file, size, params) VALUES ($1, $2, $3, $4)", type, file, size, params)
    await conn.close()
    return {"message": "Training dataset saved successfully"}

async def get_from_db():
    conn = await asyncpg.connect(DB)
    fetched_rows = []  # To store fetched rows

    try:
        async with conn.transaction():
            rows = await conn.fetch("SELECT * FROM training_queue")
            for row in rows:
                fetched_rows.append(dict(row))

    finally:
        await conn.close()
    return fetched_rows