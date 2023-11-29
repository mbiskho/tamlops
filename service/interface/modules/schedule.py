from modules.database import get_from_db
import pickle
import pandas as pd
import numpy as np
import json 
from modules.request import send_get_request, send_post_request
from modules.algo import allocate_gpu

async def schedule_logic():
    tasks = await get_from_db()

    # Load the linear regression model from the .pkl file
    with open('./models/text_scheduler_gpu_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # List to store tasks with their estimated times
    tasks_with_times = []

    check_gpu = await send_get_request("http://127.0.0.1:5000/check-gpu")

    for task in tasks:
        # Parse the JSON string to a dictionary
        params_dict = json.loads(task['params'])

        new_features = pd.DataFrame({
            'num_train_epochs': [params_dict['num_train_epochs']],
            'learning_rate': [params_dict['learning_rate']],
            'per_device_eval_batch_size': [params_dict['per_device_eval_batch_size']],
            'per_device_train_batch_size': [params_dict['per_device_train_batch_size']],
            'file_size (bytes)': [task['size']]
        })

        # Use the loaded model to predict the target variable for the new feature values
        predicted_time_raw = model.predict(new_features)
        
        predicted_metric = np.maximum(predicted_time_raw, 0)

        # Append task with its predicted time to the list and restructure it
        tasks_with_times.append({
            "file": task.file,
            "type": task.type,
            "num_gpu": 1,
            "id": task.id,
            "estimated_time": predicted_metric[0][0],
            "gpu_usage": predicted_metric[0][1],
            "param": {
                "per_device_train_batch_size": params_dict['per_device_train_batch_size'],
                "per_device_eval_batch_size": params_dict['per_device_eval_batch_size'],
                "learning_rate": params_dict['learning_rate'],
                "num_train_epochs": params_dict['num_train_epochs']
            }
        })

    # Sort tasks based on estimated time (from least to longest)
    sorted_tasks = sorted(tasks_with_times, key=lambda x: x['estimated_time'])

    # Allocate it to the right GPU
    allocated_tasks = allocate_gpu(sorted_tasks, check_gpu.response)

    # Send to DGX
    post_response = await send_post_request("http://127.0.0.1:5000/train", allocated_tasks)

    return sorted_tasks, check_gpu, post_response