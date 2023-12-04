import asyncpg
import os
import json

DB = "postgres://postgres.obxofwxgpggksbvuzzfn:tamlops123!@aws-0-us-east-1.pooler.supabase.com:6543/postgres"

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

async def delete_all_from_table():
    print(DB)
    conn = await asyncpg.connect(DB)
    try:
        await conn.execute("DELETE FROM training_queue")  # Replace 'your_table_name' with your actual table name
        return {"message": "All items deleted from the table successfully"}
    except asyncpg.PostgresError as e:
        return {"error": f"Error deleting items: {e}"}
    finally:
        await conn.close()
        return {"message": "Data has been deleted"}