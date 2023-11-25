import requests

def send_get_request(url, params=None):
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