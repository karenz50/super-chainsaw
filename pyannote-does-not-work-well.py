# log in to huggingface
from dotenv import load_dotenv
import os
from huggingface_hub import login

load_dotenv()

login(token=os.getenv("HUGGING_FACE_TOKEN"))

# audio diarization
from pyannote.audio import Pipeline
import pathlib

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

# run on a sample .wav
wav_file = pathlib.Path("data/Julian_plix5ba068d6-shift-20250616_085243.wav")
diarization = pipeline(str(wav_file))

print("Speaker Turns")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}s - {turn.end:.1f}s: speaker {speaker}")

# Check for overlapping segments
print("\nOverlapping Speech")
overlaps = diarization.get_overlap()
if overlaps:
    for overlap in overlaps.support():
        speakers = diarization.crop(overlap).labels()
        print(f"{overlap.start:.1f}s - {overlap.end:.1f}s: overlap between {', '.join(speakers)}")
else:
    print("No overlapping speech detected.")