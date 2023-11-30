import subprocess
import json

def get_gpu_info():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        output = result.stdout.strip()

        gpu_info = []
        lines = output.split('\n')
        for line in lines:
            gpu_data = line.split(',')
            gpu_info.append({
                'index': int(gpu_data[0]),
                'uuid': gpu_data[1],
                'name': gpu_data[2],
                'memory_total': int(gpu_data[3]),
                'memory_used': int(gpu_data[4]),
                'memory_free': int(gpu_data[5]),
                'utilization_gpu': int(gpu_data[6]),
                'utilization_memory': int(gpu_data[7])
            })

        return gpu_info
    except Exception as e:
        print(f"Error occurred: {e}")
        return None