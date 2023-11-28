from modules.database import get_from_db
import pickle
import pandas as pd
import numpy as np
import json 
from modules.request import send_get_request

async def schedule_logic():
    tasks = await get_from_db()

    # Load the linear regression model from the .pkl file
    with open('./models/text_scheduler_model.pkl', 'rb') as file:
        model = pickle.load(file) 

    # List to store tasks with their estimated times
    tasks_with_times = []

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
        
        predicted_time = np.maximum(predicted_time_raw, 0)

         # Append task with its predicted time to the list
        tasks_with_times.append({'task': task, 'estimated_time': predicted_time[0]})

    # Sort tasks based on estimated time (from least to longest)
    sorted_tasks = sorted(tasks_with_times, key=lambda x: x['estimated_time'])
    print(sorted_tasks)

    check_gpu = send_get_request("http://127.0.0.1:5000/check-gpu")

    return sorted_tasks, check_gpu