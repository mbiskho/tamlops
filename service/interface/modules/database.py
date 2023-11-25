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

    try:
        async with conn.transaction():
            # Execute the query and fetch results
            rows = await conn.fetch("SELECT * FROM training_queue")
            for row in rows:
                print(row)

    finally:
        await conn.close()
    return {"message": "Get all command success"}