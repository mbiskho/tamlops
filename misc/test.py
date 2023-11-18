# # import torch
# import time
# import csv
# import os
# import psutil

# cpu_usage = psutil.cpu_percent(interval=1)
# print(f"CPU Usage: {cpu_usage}%")

# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # gpu_usage = torch.cuda.memory_allocated(device) / (1024 ** 3)
# # gpu_memory = torch.cuda.memory_reserved(device) / (1024 ** 3)
# # print(f"GPU Usage: {gpu_usage:.2f}GB / {gpu_memory:.2f}GB")

# # Function to write data to CSV file
# def write_to_csv(data, csv_file_path):
#     file_exists = os.path.isfile(csv_file_path)

#     with open(csv_file_path, mode='a', newline='') as file:
#         fieldnames = list(data.keys())
#         writer = csv.DictWriter(file, fieldnames=fieldnames)

#         # If the file doesn't exist, write the header
#         if not file_exists:
#             writer.writeheader()

#         # Write the data to the CSV file
#         writer.writerow(data)

# # Example data (you can replace this with data from your requests)
# to_logs = {
#     'train_execution': end_trainning,
#     'overall_execution': end_all,
#     'cpu_usage': cpu_usage,
#     'gpu_usage': gpu_usage,
#     'resolution': data['param']['num_train_epochs'],
#     'train_batch_size': data['param']['learning_rate'],
#     'num_train_epochs': data['param']['num_train_epochs'],
#     'max_train_steps': data['param']['max_train_steps'],
#     'learning_rate': data['param']['learning_rate'],
#     'gradient_accumulation_steps':  data['param']['gradient_accumulation_steps'],
#     'num_dataset': data['param']['num_dataset']
# } 



# # Write the new data to the CSV file
# write_to_csv(data, 'text.csv')
