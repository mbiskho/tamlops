from redis import Redis
import json

# Connect to Redis
r = Redis(host='redis-13388.c292.ap-southeast-1-1.ec2.cloud.redislabs.com', port=13388,password='AuVQAhhAkVRCg3g6yQXD43VNZ0IYNZ67')


def set_item(key, value):
    r.set(key, value)

def get_item(key):
    value = r.get(key)
    return value.decode('utf-8') if value else None
