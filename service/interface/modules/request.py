import requests
import json

async def send_get_request(url, params=None):
    try:
        response = requests.get(url, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Error during GET request: {e}")
        return None

async def send_post_request(url, data):
    try:
        response = requests.post(url, data=data)
        print(response)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Error during POST request: {e}")
        return None
    
async def send_check_gpu():
    try:
        check_gpu_alfa = await requests.get("http://127.0.0.1:5000/check-gpu")
        check_gpu_beta = await requests.get("http://127.0.0.1:5000/check-gpu")
        gpu_alfa_list = json.loads(check_gpu_alfa['response'])
        gpu_beta_list = json.loads(check_gpu_beta['response'])
        combined_check_gpu = gpu_alfa_list + gpu_beta_list
        return combined_check_gpu

    except requests.RequestException as e:
        print(f"Error during GET request: {e}")
        return None