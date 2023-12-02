import aiohttp
import asyncio
import json

async def send_post_request(session, url, payload, headers):
    async with session.post(url, headers=headers, data=json.load(payload)) as response:
        pass  # No need to process or wait for the response

async def send_multiple_post_requests():
    urls = [
        'http://127.0.0.1:6060/train',  # Replace with your URLs
        'http://127.0.0.1:6060/train',
        'http://127.0.0.1:6060/train'
    ]

    payloads = [
       {"data": {"id": 14, "gpu": "3", "type": "text", "file": "https://storage.googleapis.com/training-dataset-tamlops/test_7eee5f14-7c32-4f0c-b7ca-e9bf61cf12a3_ea3f89d6-45ab-4075-acc8-84d02f934872.json", "param": {"per_device_train_batch_size": 2, "per_device_eval_batch_size": 2, "learning_rate": 0.001, "num_train_epochs": 10}}}, 
       {"data": {"id": 15, "gpu": "3", "type": "text", "file": "https://storage.googleapis.com/training-dataset-tamlops/test_7eee5f14-7c32-4f0c-b7ca-e9bf61cf12a3_ea3f89d6-45ab-4075-acc8-84d02f934872.json", "param": {"per_device_train_batch_size": 2, "per_device_eval_batch_size": 2, "learning_rate": 0.001, "num_train_epochs": 10}}},
       {"data": {"id": 16, "gpu": "3", "type": "text", "file": "https://storage.googleapis.com/training-dataset-tamlops/test_7eee5f14-7c32-4f0c-b7ca-e9bf61cf12a3_ea3f89d6-45ab-4075-acc8-84d02f934872.json", "param": {"per_device_train_batch_size": 2, "per_device_eval_batch_size": 2, "learning_rate": 0.001, "num_train_epochs": 10}}}
    ]

    headers = {'Content-Type': 'application/json'}

    async with aiohttp.ClientSession() as session:
        # List to store individual task coroutines
        post_requests = []

        for url, payload in zip(urls, payloads):
            post_requests.append(send_post_request(session, url, json.dumps(payload), headers))

        # Run all POST requests concurrently using asyncio.gather()
        await asyncio.gather(*post_requests)

# Run the function to send multiple POST requests concurrently
asyncio.run(send_multiple_post_requests())