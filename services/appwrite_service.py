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
