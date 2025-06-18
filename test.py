import os
from dotenv import load_dotenv
from azure.cosmos import CosmosClient
from azure.storage.blob import BlobServiceClient, ContentSettings
import pathlib
import urllib.parse
import subprocess
import tempfile

load_dotenv()

# load env variables
endpoint = os.getenv("COSMOS_DB_ENDPOINT")
key = os.getenv("COSMOS_KEY")
video_db_name = os.getenv("VIDEO_DB_NAME")
video_container_name = os.getenv("VIDEO_CONTAINER_NAME")
storage_cs = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
# orgs_db_name = os.getenv("ORGS_DB_NAME")
# orgs_container_name = os.getenv("ORGS_CONTAINER_NAME")

TARGET_USER_ID = "auth0|66623577e8564b8fb0940504"
DOWNLOAD_DIR = pathlib.Path("data") # local folder
FFMPEG_PATH = "tools/ffmpeg"
SIZE = 50

# cosmos query
client = CosmosClient(endpoint, credential=key)

container = client.get_database_client(video_db_name).get_container_client(video_container_name)
# orgs_container = client.get_database_client(orgs_db_name).get_container_client(orgs_container_name)

query = "SELECT * FROM c WHERE c.userId = @userId ORDER BY c.uploadDate DESC"

items = list(container.query_items(
    query=query,
    parameters=[{"name": "@userId", "value": TARGET_USER_ID}],
    enable_cross_partition_query=True
))[:SIZE]

print(f"{len(items)} items downloaded for {TARGET_USER_ID}")

# blob
blob_service = BlobServiceClient.from_connection_string(storage_cs)
DOWNLOAD_DIR.mkdir(exist_ok=True) # ensure directory exists

def download_blob_audio(blob_url):
    parsed = urllib.parse.urlparse(blob_url)
    container_name, blob_name = parsed.path.lstrip("/").split("/", 1)
    blob_client = blob_service.get_blob_client(container_name, blob_name)

    local_audio = DOWNLOAD_DIR / pathlib.Path(blob_name).with_suffix(".mp3").name
    print(f"streaming audio to {local_audio}")

    stream = blob_client.download_blob()

    with tempfile.TemporaryFile() as temp_input:
        stream.readinto(temp_input)
        temp_input.seek(0)
        subprocess.run(
            [
                FFMPEG_PATH,
                "-y", # overwrite output if exists
                "-i", "pipe:0", # read input from stdin
                "-vn", # no video
                "-acodec", "libmp3lame",
                str(local_audio)
            ],
            stdin=temp_input,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )

# iterate and download
for item in items:
    blob_url = item.get("BlobPath")
    if not blob_url:
        print(f"no blobPath found in document id={item.get('id')}, skipping.")
        continue
    try:
        download_blob_audio(blob_url)
    except Exception as exc:
        print(f"failed for {blob_url}: {exc}")

print("complete")