from modules.database import get_from_db
import pickle
import pandas as pd
import numpy as np
import json 
from modules.request import send_get_request, send_post_request, send_check_gpu
from modules.algo import allocate_gpu
from modules.check_redis import get_redis_item
import time
import requests
import asyncio
import aiohttp

async def schedule_logic_min_min():
    tasks = await get_from_db()

    # Load the linear regression model from the .pkl file
    with open('./models/text_to_text_model.pkl', 'rb') as file:
        model_text = pickle.load(file)

    # Load the linear regression model from the .pkl file
    with open('./models/text_to_image_model.pkl', 'rb') as file:
        model_image = pickle.load(file)

    # List to store tasks with their estimated times
    tasks_with_times = []

    for task in tasks:
        # Parse the JSON string to a dictionary
        params_dict = json.loads(task['params'])

        if task['type'] == 'text':
            new_features = pd.DataFrame({
                'num_train_epochs': [params_dict['num_train_epochs']],
                'learning_rate': [params_dict['learning_rate']],
                'per_device_eval_batch_size': [params_dict['per_device_eval_batch_size']],
                'per_device_train_batch_size': [params_dict['per_device_train_batch_size']],
                'file_size (bytes)': [task['size']]
            })

            # Use the loaded model to predict the target variable for the new feature values
            predicted_time_raw = model_text.predict(new_features)
            
            predicted_metric = np.maximum(predicted_time_raw, 0)

            # Append task with its predicted time to the list and restructure it
            tasks_with_times.append({
                "file": task['file'],
                "type": task['type'],
                "id": task['id'],
                "estimated_time": predicted_metric[0][0],
                "gpu_usage": predicted_metric[0][1],
                "param": {
                    "per_device_train_batch_size": params_dict['per_device_train_batch_size'],
                    "per_device_eval_batch_size": params_dict['per_device_eval_batch_size'],
                    "learning_rate": params_dict['learning_rate'],
                    "num_train_epochs": params_dict['num_train_epochs']
                }
            })
        elif task['type'] == 'image':
            new_features = pd.DataFrame({
                'resolution': params_dict['resolution'],
                'train_batch_size': params_dict['train_batch_size'],
                'num_train_epochs': params_dict['num_train_epochs'],
                'max_train_steps': params_dict['max_train_steps'],
                'learning_rate': params_dict['learning_rate'],
                'gradient_accumulation_steps': params_dict['gradient_accumulation_steps'],
                'file_size': params_dict['file_size']
            })

             # Use the loaded model to predict the target variable for the new feature values
            predicted_time_raw = model_image.predict(new_features)
            
            predicted_metric = np.maximum(predicted_time_raw, 0)

             # Append task with its predicted time to the list and restructure it
            tasks_with_times.append({
                    "file": task['file'],
                    "type": task['type'],
                    "id": task['id'],
                    "estimated_time": predicted_metric[0][0],
                    "gpu_usage": predicted_metric[0][1],
                    "param": {
                        'resolution': params_dict['resolution'],
                        'train_batch_size': params_dict['train_batch_size'],
                        'num_train_epochs': params_dict['num_train_epochs'],
                        'max_train_steps': params_dict['max_train_steps'],
                        'learning_rate': params_dict['learning_rate'],
                        'gradient_accumulation_steps': params_dict['gradient_accumulation_steps'],
                    }
            })


    # Sort tasks based on estimated time (from least to longest)
    sorted_tasks = sorted(tasks_with_times, key=lambda x: x['estimated_time'])

    #Check GPU
    dgx_gpu = await send_get_request('http://127.0.0.1:6060/')

    # Allocate it to the right GPU
    allocated_tasks = allocate_gpu(sorted_tasks, dgx_gpu['response'])

    # Send to DGX
    for task in allocated_tasks:
        check_gpu = await send_get_request('http://127.0.0.1:6060/')
        current_gpu_state = check_gpu['response']
        current_free_memory = 0
        for gpu in current_gpu_state:
             if gpu['index'] == task['num_gpu']:
                 current_free_memory = gpu['memory_free']
        if current_free_memory > task['gpu_usage']:
            send_post_request("http://127.0.0.1:6060/train", {"data": task})
        else:     
            finished_flag = False
            while finished_flag == False:
                redis_value = get_redis_item(gpu['index'])
                if redis_value == 0:
                    send_post_request("http://127.0.0.1:6060/train", {"data": task})
                    finished_flag = True
                    break
                time.sleep(5)

    return {"error": False, "response": "Scheduling Finished"}

async def schedule_logic_max_min():
    return 1

async def send_post_request_async(session, url, payload, headers):
    print(payload)
    print(json.loads(payload))

    async with session.post(url, headers=headers, data=json.loads(payload)) as response:
        if response.status == 200:
            print(f"Sent Success")

async def schedule_logic_fcfs():
    tasks = await get_from_db()
    print(tasks)
    
    async with aiohttp.ClientSession() as session:
        # List to store individual task coroutines
        post_requests = []

        for task in tasks:
            print(task)
            params_dict = json.loads(task['params'])
            print(task['id'])
            print(params_dict['per_device_train_batch_size'])

            url = "https://webhook.site/867e52b5-ecce-4737-a94d-9b90c6526f46"

            payload = json.dumps({
            "data": {
                "id": task['id'],
                "gpu": "3",
                "type": task['type'],
                "file": task['file'],
                "param": {
                "per_device_train_batch_size": params_dict['per_device_train_batch_size'],
                "per_device_eval_batch_size": params_dict['per_device_eval_batch_size'],
                "learning_rate": params_dict['learning_rate'],
                "num_train_epochs": params_dict['num_train_epochs']
                }
            }
            })
            headers = {
            'Content-Type': 'application/json'
            }

            post_requests.append(send_post_request_async(session, url, json.dumps(payload), headers))
    
        # Run all POST requests concurrently using asyncio.gather()
        await asyncio.gather(*post_requests)

    return {"error": False, "response": "Scheduling Finished"}