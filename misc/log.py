import torch
import time
import csv
import os
import psutil

cpu_usage = psutil.cpu_percent(interval=1)
print(f"CPU Usage: {cpu_usage}%")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_usage = torch.cuda.memory_allocated(device) / (1024 ** 3)
gpu_memory = torch.cuda.memory_reserved(device) / (1024 ** 3)
print(device)
print(f"GPU Usage: {gpu_usage:.2f}GB / {gpu_memory:.2f}GB")
