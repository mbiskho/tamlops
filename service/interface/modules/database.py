import asyncpg
import os
import json

DB = "postgres://postgres:pass@localhost:5432/postgres"

async def save_training_db(type, file, size, params):
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
    
async def delete_row_by_id(table_name, row_id):
    # Establish a connection to the PostgreSQL database
    conn = await asyncpg.connect(DB)
    try:
        # Construct the DELETE query using the provided table name and ID
        delete_query = f"DELETE FROM {table_name} WHERE id = $1;"
        
        # Execute the DELETE query with the specified ID
        await conn.execute(delete_query, row_id)
        print(f"Row with ID {row_id} deleted successfully!")

    except asyncpg.PostgresError as e:
        print("Error:", e)
    
    finally:
        # Close the connection
        if conn:
            await conn.close()