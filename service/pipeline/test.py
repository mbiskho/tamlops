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

result = subprocess.run(command, capture_output=True, text=True, check=True)
