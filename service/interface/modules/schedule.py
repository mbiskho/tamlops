from modules.database import get_from_db, delete_row_by_id
import pickle
import pandas as pd
import numpy as np
import json 
from modules.request import send_get_request, send_post_request, send_check_gpu
from modules.algo import allocate_gpu, real_min_min
from modules.check_redis import get_redis_item
import time
import requests
import asyncio
import aiohttp
import threading


async def schedule_logic_real_min_min():
    tasks = await get_from_db()
    # print("Tasks from DB", tasks)
    if tasks == []:
        return {"error": True, "response": "There are no data in Queue"} 

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
        # print("Single Task", task)

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
                "gpu_usage": 3000,
                "param": {
                    "per_device_train_batch_size": params_dict['per_device_train_batch_size'],
                    "per_device_eval_batch_size": params_dict['per_device_eval_batch_size'],
                    "learning_rate": params_dict['learning_rate'],
                    "num_train_epochs": params_dict['num_train_epochs']
                }
            })
        elif task['type'] == 'image':
            new_features = pd.DataFrame({
                'resolution': [params_dict['resolution']],
                'train_batch_size': [params_dict['train_batch_size']],
                'num_train_epochs': [params_dict['num_train_epochs']],
                'max_train_steps': [params_dict['max_train_steps']],
                'learning_rate': [params_dict['learning_rate']],
                'gradient_accumulation_steps': [params_dict['gradient_accumulation_steps']],
                'file_size': [task['size']]
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
                    "gpu_usage": 15000,
                    "param": {
                        'resolution': params_dict['resolution'],
                        'train_batch_size': params_dict['train_batch_size'],
                        'num_train_epochs': params_dict['num_train_epochs'],
                        'max_train_steps': params_dict['max_train_steps'],
                        'learning_rate': params_dict['learning_rate'],
                        'gradient_accumulation_steps': params_dict['gradient_accumulation_steps'],
                    }
            })


    # # Sort tasks based on estimated time (from least to longest)
    # sorted_tasks = sorted(tasks_with_times, key=lambda x: x['estimated_time'])

    #Check GPU
    dgx_gpu = await send_get_request('http://127.0.0.1:6070/check-gpu')

    #Sort task using Real Min-Min Algorithm
    sorted_tasks = real_min_min(tasks_with_times, dgx_gpu)

    # Allocate it to the right GPU
    allocated_tasks = allocate_gpu(sorted_tasks, dgx_gpu['response'])

    url = "http://127.0.0.1:6070/train-burst"
    headers = {
    'Content-Type': 'application/json'
    }

    # print("Allocated Tasks", allocated_tasks)

    # Send to DGX
    for task in allocated_tasks:
        check_gpu = await send_get_request('http://127.0.0.1:6070/check-gpu')
        current_gpu_state = check_gpu['response']
        # print("All GPU State", current_gpu_state)
        current_free_memory = current_gpu_state[task['gpu']]['memory_free']
        print(f"Current GPU {task['gpu']} Free Memory", current_free_memory)
        print(f"Task ID: {task['id']} and Estimated Task GPU {task['gpu']} Usage", task['gpu_usage'])
        if current_free_memory > task['gpu_usage']:
            del task['gpu_usage']
            response = requests.request("POST", url, headers=headers, data=json.dumps({
            "data": task
            }))
            print("POST Response", response)
            await delete_row_by_id("training_queue", task['id'])
        else:
            print("[!] GPU Memory Full")     
            finished_flag = False
            while finished_flag == False:
                redis_value = get_redis_item(task['gpu'])
                print("Redis Value:" , redis_value)
                print("Redis Value Datatype:", type(redis_value))
                print("Redis Check", redis_value == "0")
                if redis_value == "0":
                    del task['gpu_usage']
                    response = requests.request("POST", url, headers=headers, data=json.dumps({
                    "data": task
                    }))
                    await delete_row_by_id("training_queue", task['id'])
                    finished_flag = True
                    break
                time.sleep(5)

    return {"error": False, "response": "Scheduling Finished"}

async def schedule_logic_min_min():
    tasks = await get_from_db()
    # print("Tasks from DB", tasks)
    if tasks == []:
        return {"error": True, "response": "There are no data in Queue"} 

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
        # print("Single Task", task)

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
                "gpu_usage": 3000,
                "param": {
                    "per_device_train_batch_size": params_dict['per_device_train_batch_size'],
                    "per_device_eval_batch_size": params_dict['per_device_eval_batch_size'],
                    "learning_rate": params_dict['learning_rate'],
                    "num_train_epochs": params_dict['num_train_epochs']
                }
            })
        elif task['type'] == 'image':
            new_features = pd.DataFrame({
                'resolution': [params_dict['resolution']],
                'train_batch_size': [params_dict['train_batch_size']],
                'num_train_epochs': [params_dict['num_train_epochs']],
                'max_train_steps': [params_dict['max_train_steps']],
                'learning_rate': [params_dict['learning_rate']],
                'gradient_accumulation_steps': [params_dict['gradient_accumulation_steps']],
                'file_size': [task['size']]
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
                    "gpu_usage": 15000,
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
    dgx_gpu = await send_get_request('http://127.0.0.1:6070/check-gpu')

    # Allocate it to the right GPU
    allocated_tasks = allocate_gpu(sorted_tasks, dgx_gpu['response'])

    url = "http://127.0.0.1:6070/train-burst"
    headers = {
    'Content-Type': 'application/json'
    }

    # print("Allocated Tasks", allocated_tasks)
    # current_gpu_state = []
    check_gpu = await send_get_request('http://127.0.0.1:6070/check-gpu')
    current_gpu_state = check_gpu['response']
    # Send to DGX
    for task in allocated_tasks:
        # print("All GPU State", current_gpu_state)
        current_free_memory = current_gpu_state[task['gpu']]['memory_free']
        print(f"Current GPU {task['gpu']} Free Memory", current_free_memory)
        print(f"Task ID: {task['id']} and Estimated Task GPU {task['gpu']} Usage", task['gpu_usage'])
        if current_free_memory > task['gpu_usage']:
            current_gpu_state[task['gpu']]['memory_free'] = current_gpu_state[task['gpu']]['memory_free'] - task['gpu_usage']
            del task['gpu_usage']
            response = requests.request("POST", url, headers=headers, data=json.dumps({
            "data": task
            }))
            print("POST Response", response)
            await delete_row_by_id("training_queue", task['id'])
        else:
            print("[!] GPU Memory Full")     
            finished_flag = False
            while finished_flag == False:
                redis_value = get_redis_item(task['gpu'])
                print("Redis Value:" , redis_value)
                print("Redis Value Datatype:", type(redis_value))
                print("Redis Check", redis_value == "0")
                if redis_value == "0":
                    del task['gpu_usage']
                    response = requests.request("POST", url, headers=headers, data=json.dumps({
                    "data": task
                    }))
                    finished_flag = True
                    check_gpu = await send_get_request('http://127.0.0.1:6070/check-gpu')
                    current_gpu_state = check_gpu['response']
                    await delete_row_by_id("training_queue", task['id'])
                    break
                time.sleep(5)

    return {"error": False, "response": "Scheduling Finished"}

async def schedule_logic_max_min():
    tasks = await get_from_db()
    if tasks == []:
        return {"error": True, "response": "There are no data in Queue"} 

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
                "gpu_usage": 3000,
                "param": {
                    "per_device_train_batch_size": params_dict['per_device_train_batch_size'],
                    "per_device_eval_batch_size": params_dict['per_device_eval_batch_size'],
                    "learning_rate": params_dict['learning_rate'],
                    "num_train_epochs": params_dict['num_train_epochs']
                }
            })
        elif task['type'] == 'image':
            new_features = pd.DataFrame({
                'resolution': [params_dict['resolution']],
                'train_batch_size': [params_dict['train_batch_size']],
                'num_train_epochs': [params_dict['num_train_epochs']],
                'max_train_steps': [params_dict['max_train_steps']],
                'learning_rate': [params_dict['learning_rate']],
                'gradient_accumulation_steps': [params_dict['gradient_accumulation_steps']],
                'file_size': [task['size']]
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
                    "gpu_usage": 15000,
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
    sorted_tasks = sorted(tasks_with_times, key=lambda x: x['estimated_time'], reverse=True)

    #Check GPU
    dgx_gpu = await send_get_request('http://127.0.0.1:6070/check-gpu')

    # Allocate it to the right GPU
    allocated_tasks = allocate_gpu(sorted_tasks, dgx_gpu['response'])

    url = "http://127.0.0.1:6070/train-burst"
    headers = {
    'Content-Type': 'application/json'
    }

    check_gpu = await send_get_request('http://127.0.0.1:6070/check-gpu')
    current_gpu_state = check_gpu['response']

    # Send to DGX
    for task in allocated_tasks:
        current_free_memory = current_gpu_state[task['gpu']]['memory_free']
        print(f"Current GPU {task['gpu']} Free Memory", current_free_memory)
        print(f"Estimated task {task['id']} GPU {task['gpu']} Usage", task['gpu_usage'])
        if current_free_memory > task['gpu_usage']:
            current_gpu_state[task['gpu']]['memory_free'] = current_gpu_state[task['gpu']]['memory_free'] - task['gpu_usage']
            del task['gpu_usage']
            response = requests.request("POST", url, headers=headers, data=json.dumps({
            "data": task
            }))
            print("POST Response", response)
            await delete_row_by_id("training_queue", task['id'])
        else:
            print("[!] GPU Memory Full")     
            finished_flag = False
            while finished_flag == False:
                redis_value = get_redis_item(task['gpu'])
                print("Redis Value:" , redis_value)
                print("Redis Check", redis_value == "0")
                if redis_value == "0":
                    del task['gpu_usage']
                    response = requests.request("POST", url, headers=headers, data=json.dumps({
                    "data": task
                    }))
                    check_gpu = await send_get_request('http://127.0.0.1:6070/check-gpu')
                    await delete_row_by_id("training_queue", task['id'])
                    current_gpu_state = check_gpu['response']
                    finished_flag = True
                    break
                time.sleep(5)
    
    return {"error": False, "response": "Scheduling Finished"}

async def send_post_request_async(session, url, payload, headers):
    print(payload)
    print(json.loads(payload))

    async with session.post(url, headers=headers, data=json.loads(payload)) as response:
        if response.status == 200:
            print(f"Sent Success")

async def schedule_logic_fcfs_burst():
    tasks = await get_from_db()
    print(tasks)

    if tasks == []:
        return {"error": True, "response": "There are no data in Queue"} 
    
    async with aiohttp.ClientSession() as session:
        # List to store individual task coroutines
        post_requests = []

        for task in tasks:
            print(task)
            params_dict = json.loads(task['params'])
            print(task['id'])

            url = "http://127.0.0.1:6070/train-burst-nogpu"

            payload = ''
            if task['type'] == 'text':
                payload = json.dumps({
                "data": {
                    "id": task['id'],
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
            elif task['type'] == 'image':
                payload = json.dumps({
                "data": {
                    "id": task['id'],
                    "type": task['type'],
                    "file": task['file'],
                    "param": {
                            'resolution': params_dict['resolution'],
                            'train_batch_size': params_dict['train_batch_size'],
                            'num_train_epochs': params_dict['num_train_epochs'],
                            'max_train_steps': params_dict['max_train_steps'],
                            'learning_rate': params_dict['learning_rate'],
                            'gradient_accumulation_steps': params_dict['gradient_accumulation_steps'],
                    }
                }
            })
            headers = {
            'Content-Type': 'application/json'
            }

            post_requests.append(send_post_request_async(session, url, json.dumps(payload), headers))
            await delete_row_by_id("training_queue", task['id'])
    
        # Run all POST requests concurrently using asyncio.gather()
        await asyncio.gather(*post_requests)

    return {"error": False, "response": "Scheduling Finished"}


async def schedule_logic_fcfs_normal():
    tasks = await get_from_db()
    print(tasks)

    if tasks == []:
        return {"error": True, "response": "There are no data in Queue"} 

    for task in tasks:
        print(task)
        params_dict = json.loads(task['params'])
        print(task['id'])

        url = "http://127.0.0.1:6070/train-nogpu"

        payload = ''
        if task['type'] == 'text':
            payload = json.dumps({
            "data": {
                "id": task['id'],
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
        elif task['type'] == 'image':
            payload = json.dumps({
            "data": {
                "id": task['id'],
                "type": task['type'],
                "file": task['file'],
                "param": {
                        'resolution': params_dict['resolution'],
                        'train_batch_size': params_dict['train_batch_size'],
                        'num_train_epochs': params_dict['num_train_epochs'],
                        'max_train_steps': params_dict['max_train_steps'],
                        'learning_rate': params_dict['learning_rate'],
                        'gradient_accumulation_steps': params_dict['gradient_accumulation_steps'],
                }
            }
        })


        headers = {
        'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        print(response)
        await delete_row_by_id("training_queue", task['id'])

    return {"error": False, "response": "Scheduling Finished"}

async def schedule_logic_fcfs_normal_gpu(gpu_id):
    tasks = await get_from_db()
    print(tasks)

    if tasks == []:
        return {"error": True, "response": "There are no data in Queue"} 

    for task in tasks:
        print(task)
        params_dict = json.loads(task['params'])
        print(task['id'])

        url = "http://127.0.0.1:6070/train"

        payload = ''
        if task['type'] == 'text':
            payload = json.dumps({
            "data": {
                "id": task['id'],
                "type": task['type'],
                "file": task['file'],
                "gpu": gpu_id,
                "param": {
                    "per_device_train_batch_size": params_dict['per_device_train_batch_size'],
                    "per_device_eval_batch_size": params_dict['per_device_eval_batch_size'],
                    "learning_rate": params_dict['learning_rate'],
                    "num_train_epochs": params_dict['num_train_epochs']
                }
            }
        })
        elif task['type'] == 'image':
            payload = json.dumps({
            "data": {
                "id": task['id'],
                "type": task['type'],
                "file": task['file'],
                "gpu": gpu_id,
                "param": {
                        'resolution': params_dict['resolution'],
                        'train_batch_size': params_dict['train_batch_size'],
                        'num_train_epochs': params_dict['num_train_epochs'],
                        'max_train_steps': params_dict['max_train_steps'],
                        'learning_rate': params_dict['learning_rate'],
                        'gradient_accumulation_steps': params_dict['gradient_accumulation_steps'],
                }
            }
        })


        headers = {
        'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        print(response)
        await delete_row_by_id("training_queue", task['id'])

    return {"error": False, "response": "Scheduling Finished"}


async def schedule_logic_fcfs_burst_gpu(gpu_id):
    tasks = await get_from_db()
    print(tasks)

    if tasks == []:
        return {"error": True, "response": "There are no data in Queue"} 
    
    async with aiohttp.ClientSession() as session:
        # List to store individual task coroutines
        post_requests = []

        for task in tasks:
            print(task)
            params_dict = json.loads(task['params'])
            print(task['id'])

            url = "http://127.0.0.1:6070/train-burst"

            payload = ''
            if task['type'] == 'text':
                payload = json.dumps({
                "data": {
                    "id": task['id'],
                    "type": task['type'],
                    "file": task['file'],
                    "gpu": gpu_id,
                    "param": {
                        "per_device_train_batch_size": params_dict['per_device_train_batch_size'],
                        "per_device_eval_batch_size": params_dict['per_device_eval_batch_size'],
                        "learning_rate": params_dict['learning_rate'],
                        "num_train_epochs": params_dict['num_train_epochs']
                    }
                }
            })
            elif task['type'] == 'image':
                payload = json.dumps({
                "data": {
                    "id": task['id'],
                    "type": task['type'],
                    "file": task['file'],
                    "gpu": gpu_id,
                    "param": {
                            'resolution': params_dict['resolution'],
                            'train_batch_size': params_dict['train_batch_size'],
                            'num_train_epochs': params_dict['num_train_epochs'],
                            'max_train_steps': params_dict['max_train_steps'],
                            'learning_rate': params_dict['learning_rate'],
                            'gradient_accumulation_steps': params_dict['gradient_accumulation_steps'],
                    }
                }
            })
            headers = {
            'Content-Type': 'application/json'
            }

            post_requests.append(send_post_request_async(session, url, json.dumps(payload), headers))
            await delete_row_by_id("training_queue", task['id'])
    
        # Run all POST requests concurrently using asyncio.gather()
        await asyncio.gather(*post_requests)

    return {"error": False, "response": "Scheduling Finished"}


