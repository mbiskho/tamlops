# import torch
# import time
# import csv
# import os
# import psutil

# cpu_usage = psutil.cpu_percent(interval=1)
# print(f"CPU Usage: {cpu_usage}%")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# gpu_usage = torch.cuda.memory_allocated(device) / (1024 ** 3)
# gpu_memory = torch.cuda.memory_reserved(device) / (1024 ** 3)
# print(device)
# print(f"GPU Usage: {gpu_usage:.2f}GB / {gpu_memory:.2f}GB")



data = {
    "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
    "param": {
      ""
      "resolution": 10, # 512
      "train_batch_size": 1, #6
      "num_train_epochs": 1, #100
      "max_train_steps": 10,
      "gradient_accumulation_steps": 1,
      "learning_rate": 0.0001,

      "num_dataset": 100
    }
}


datas = [
    # ------
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 10, # 512
                "train_batch_size": 1, #6
                "num_train_epochs": 1, #100
                "max_train_steps": 10,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 10
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 10, # 512
                "train_batch_size": 1, #6
                "num_train_epochs": 1, #100
                "max_train_steps": 10,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 50
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 10, # 512
                "train_batch_size": 1, #6
                "num_train_epochs": 1, #100
                "max_train_steps": 10,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 100
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 10, # 512
                "train_batch_size": 1, #6
                "num_train_epochs": 1, #100
                "max_train_steps": 10,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 200
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 10, # 512
                "train_batch_size": 1, #6
                "num_train_epochs": 1, #100
                "max_train_steps": 10,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 400
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 10, # 512
                "train_batch_size": 1, #6
                "num_train_epochs": 1, #100
                "max_train_steps": 10,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 800
        }
    },

    # ------
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 50, # 512
                "train_batch_size": 3, #6
                "num_train_epochs": 3, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 10
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 50, # 512
                "train_batch_size": 3, #6
                "num_train_epochs": 3, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 50
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 50, # 512
                "train_batch_size": 3, #6
                "num_train_epochs": 3, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 100
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 50, # 512
                "train_batch_size": 3, #6
                "num_train_epochs": 3, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 200
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 50, # 512
                "train_batch_size": 3, #6
                "num_train_epochs": 3, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 400
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 10, # 512
                "train_batch_size": 3, #6
                "num_train_epochs": 3, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 800
        }
    },
    # ------
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 256, # 512
                "train_batch_size": 6, #6
                "num_train_epochs": 6, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 10
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 256, # 512
                "train_batch_size": 6, #6
                "num_train_epochs": 6, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 50
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 256, # 512
                "train_batch_size": 6, #6
                "num_train_epochs": 6, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 100
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 256, # 512
                "train_batch_size": 6, #6
                "num_train_epochs": 6, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 200
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 256, # 512
                "train_batch_size": 6, #6
                "num_train_epochs": 6, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 400
        }
    },
    {
        "file": "https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet",
        "param": {
                "resolution": 256, # 512
                "train_batch_size": 6, #6
                "num_train_epochs": 6, #100
                "max_train_steps": 100,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.0001,

                "num_dataset": 800
        }
    },

]


for x in datas:
    print(x)