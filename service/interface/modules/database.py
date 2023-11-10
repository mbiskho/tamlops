import asyncpg
from dotenv import load_dotenv
import os

load_dotenv()

DB = os.environ.get("DB")

async def text_train_db(req):
    conn = await asyncpg.connect(DB)
    for i in req:
        await conn.execute("INSERT INTO text_train (prompt, answer) VALUES ($1, $2)", i['prompt'], i['answer'])
    await conn.close()
    return {"message": "Training dataset saved successfully"}

async def text_test_db(req):
    conn = await asyncpg.connect(DB)
    for i in req:
        await conn.execute("INSERT INTO text_test (prompt, answer) VALUES ($1, $2)", i['prompt'], i['answer'])
    await conn.close()
    return {"message": "Test dataset saved successfully"}

async def check_table_count(table_name):
    conn = await asyncpg.connect(DB)
    result = await conn.fetchval(f'SELECT COUNT(*) FROM {table_name}')
    await conn.close()
    return result
