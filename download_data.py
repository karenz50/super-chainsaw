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
SIZE = 30
MIN_DURATION = 6
TO_WAV = True

# cosmos query
client = CosmosClient(endpoint, credential=key)

container = client.get_database_client(video_db_name).get_container_client(video_container_name)
# orgs_container = client.get_database_client(orgs_db_name).get_container_client(orgs_container_name)

query = """
SELECT * FROM c
WHERE c.userId = @userId
  AND c.generatedEmbeddings = true
  AND c.watermarkedVideo = true
  AND c.assignedUserName = "DeMarcus"
ORDER BY c.uploadDate DESC
"""
# names include Anthony, Julian, DeMarcus, Casey, Chaz

items = list(container.query_items(
    query=query,
    parameters=[{"name": "@userId", "value": TARGET_USER_ID}],
    enable_cross_partition_query=True
))

print(f"{len(items)} items downloaded for {TARGET_USER_ID}")

# blob
blob_service = BlobServiceClient.from_connection_string(storage_cs)
DOWNLOAD_DIR.mkdir(exist_ok=True) # ensure directory exists

def download_blob(blob_url, to_wav=False, prefix=""):
    parsed = urllib.parse.urlparse(blob_url)
    if not parsed.scheme.startswith("http"):
        raise ValueError(f"Expected full https://... blob URL, got: {blob_url}")

    container, blob_name = parsed.path.lstrip("/").split("/", 1)
    blob_client = blob_service.get_blob_client(container, blob_name)

    # check audio duration
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_file:
        stream = blob_client.download_blob()
        stream.readinto(temp_file)
        temp_file.flush()
        result = subprocess.run(
            [FFMPEG_PATH.replace("ffmpeg", "ffprobe"),
                "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                temp_file.name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            universal_newlines=True
        )
        duration = float(result.stdout.strip())

        # skip short videos
        if duration < MIN_DURATION:
            print(f"skipping short video ({duration:.1f}s) {blob_name}")
            return None
        
        # extract audio
        ext = ".wav" if to_wav else ".mp3"
        output_name = f"{prefix}_{pathlib.Path(blob_name).stem}{ext}"
        local_audio = DOWNLOAD_DIR / output_name

        print(f"extracting audio from {blob_name} to {local_audio}")

        ffmpeg_cmd = [FFMPEG_PATH, "-y", "-i", temp_file.name, "-vn"]
        if to_wav:
            ffmpeg_cmd += ["-ac", "1", "-ar", "16000", str(local_audio)]
        else:
            ffmpeg_cmd += ["-acodec", "libmp3lame", str(local_audio)]

        subprocess.run(
            ffmpeg_cmd,
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )

        return local_audio

# iterate and download
downloaded = 0

for item in items:
    if downloaded >= SIZE:
        break

    blob_url = item.get("BlobPath")
    assigned_name = item.get("assignedUserName", "unknownuser").replace(" ", "_")

    try:
        result = download_blob(blob_url, TO_WAV, assigned_name)
        if result:
            downloaded += 1
            print(f"downloaded {downloaded} of {SIZE}")
    except Exception as exc:
        print(f"failed for {blob_url}: {exc}")

print("complete")