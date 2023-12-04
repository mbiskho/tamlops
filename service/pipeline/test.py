import subprocess

command = [
            'sh',
            'text.sh',
            f'2',
            f'1', 
            f'1', 
            f'0.0001', 
            f'4', 
            f'https://storage.googleapis.com/training-dataset-tamlops/small.json', 
            f'8012'
]

# command = [
#             'sh', 
#             'image.sh', 
#             f'1'
#             f'10', 
#             f'1', 
#             f'1', 
#             f'2', 
#             f'0.001', 
#             f'1', 
#             f'https://storage.googleapis.com/training-dataset-tamlops/train-00000-of-00001-566cc9b19d7203f8.parquet',
#             f'11293'
# ]

result = subprocess.run(command)


