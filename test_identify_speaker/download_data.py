import os
import subprocess
import pathlib
import tempfile

DATA_DIR = pathlib.Path("data")
FFMPEG_PATH = "tools/ffmpeg" 
MIN_DURATION = 6 
TO_WAV = True  # change to False to output .mp3

def convert_mp4_to_audio(mp4_path, to_wav=True):
    try:
        result = subprocess.run(
            [FFMPEG_PATH.replace("ffmpeg", "ffprobe"),
             "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(mp4_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            universal_newlines=True
        )
        duration = float(result.stdout.strip())
        if duration < MIN_DURATION:
            print(f"Skipping short file ({duration:.1f}s): {mp4_path.name}")
            return

        ext = ".wav" if to_wav else ".mp3"
        out_path = mp4_path.with_suffix(ext)

        cmd = [FFMPEG_PATH, "-y", "-i", str(mp4_path), "-vn"]
        if to_wav:
            cmd += ["-ac", "1", "-ar", "16000", str(out_path)]
        else:
            cmd += ["-acodec", "libmp3lame", str(out_path)]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"Converted {mp4_path.name} -> {out_path.name}")

    except Exception as e:
        print(f"Error converting {mp4_path.name}: {e}")

# run conversion on all .mp4 files in data/
for mp4_file in DATA_DIR.glob("*.mp4"):
    print(mp4_file)
    convert_mp4_to_audio(mp4_file, TO_WAV)
