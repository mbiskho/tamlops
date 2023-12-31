from google.cloud import storage
import os
from google.oauth2 import service_account
import uuid

def get_gcs_url(bucket_name, object_path):
    return f"https://storage.googleapis.com/{bucket_name}/{object_path}"

async def upload_to_gcs(file):
    current_directory = os.getcwd()

    credentials = service_account.Credentials.from_service_account_file(
       f'{current_directory}/modules/mlops-398205-ae400302eb1c.json', scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    bucket_name = "training-dataset-tamlops"

    name, extension = os.path.splitext(file.filename)
    unique_id = str(uuid.uuid4())
    destination_blob_name = f"{name}_{unique_id}{extension}"

    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_file(file.file)

    file_url = get_gcs_url(bucket_name, destination_blob_name)

    return file_url