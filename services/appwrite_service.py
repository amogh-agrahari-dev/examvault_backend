from appwrite.client import Client
from appwrite.services.storage import Storage
from appwrite.input_file import InputFile
from config import settings

client = Client()
client.set_endpoint(settings.APPWRITE_ENDPOINT)
client.set_project(settings.APPWRITE_PROJECT_ID)
client.set_key(settings.APPWRITE_API_KEY)

storage = Storage(client)

def upload_file_to_appwrite(file_path: str, filename: str) -> str:
    """Uploads a local file to Appwrite storage and returns the file ID."""
    try:
        result = storage.create_file(
            bucket_id=settings.APPWRITE_BUCKET_ID,
            file_id='unique()',
            file=InputFile.from_path(file_path)
        )
        return result.id
    except Exception as e:
        print(f"Error uploading to Appwrite: {e}")
        raise

def download_file_from_appwrite(file_id: str, download_path: str):
    """Downloads a file from Appwrite storage to a local path."""
    try:
        result = storage.get_file_download(
            bucket_id=settings.APPWRITE_BUCKET_ID,
            file_id=file_id
        )
        with open(download_path, "wb") as f:
            f.write(result)
        return download_path
    except Exception as e:
        print(f"Error downloading from Appwrite: {e}")
        raise

def delete_file_from_appwrite(file_id: str):
    """Deletes a file from Appwrite storage."""
    try:
        storage.delete_file(
            bucket_id=settings.APPWRITE_BUCKET_ID,
            file_id=file_id
        )
    except Exception as e:
        print(f"Error deleting from Appwrite: {e}")

