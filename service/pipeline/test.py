import subprocess

command = [
            'sh', 
            'text.sh', 
            f'5'
            f'1', 
            f'1', 
            f'0.0001', 
            f'10', 
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

result = subprocess.run(command, capture_output=True, text=True, check=True)
print("Subprocess output (stdout):", result.stdout)
print("Subprocess output (stderr):", result.stderr)


