import aiohttp
import asyncio
import requests
import json

async def text():
    url = "http://127.0.0.1:8000"
    payload = json.dumps({
    "data": {
        "id": 100,
        "gpu": "3",
        "type": "text",
        "file": "https://storage.googleapis.com/training-dataset-tamlops/test_3ce50023-e1ba-4f89-984f-9a814a374411.json",
        "param": {
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "learning_rate": 0.001,
        "num_train_epochs": 1
        }
    }
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)


async def image():
    url = "http://127.0.0.1:8000"
    payload = json.dumps({
    "data": {
        "id": 1,
        "gpu": "3",
        "type": "image",
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
        "resolution": 10,
        "train_batch_size": 2,
        "num_train_epochs": 1,
        "max_train_steps": 1,
        "gradient_accumulation_steps": 2,
        "learning_rate": 0.001
        }
    }
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)


async def main():
    # url = "http://127.0.0.1:8000"

    tasks = [text() for _ in range(3)]  # Example: Make 10 concurrent requests
    results = await asyncio.gather(*tasks)
    for result in results:
        print(result)

    tasks = [image() for _ in range(3)]  # Example: Make 10 concurrent requests
    results = await asyncio.gather(*tasks)
    for result in results:
        print(result)

if __name__ == "__main__":
    asyncio.run(main())

