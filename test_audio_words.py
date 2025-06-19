import os
from dotenv import load_dotenv
from huggingface_hub import login
import whisperx
import torch
from whisperx.diarize import DiarizationPipeline
import pandas as pd

# Load environment variables
load_dotenv()
login(token=os.getenv("HUGGING_FACE_TOKEN"))

# File path and device
AUDIO_PATH = "data/Anthony_Filamor_plix5ba068ee-shift-20250618_144548.wav"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load WhisperX ASR model
asr_model = whisperx.load_model("base.en", device, compute_type="int8")
transcription = asr_model.transcribe(AUDIO_PATH)

# Align words
model_a, metadata = whisperx.load_align_model(language_code=transcription["language"], device=device)
aligned = whisperx.align(transcription["segments"], model_a, metadata, AUDIO_PATH, device)

# Diarization
diarize_model = DiarizationPipeline(
    use_auth_token=os.getenv("HUGGING_FACE_TOKEN"),
    device=device
)
diarize_segments = diarize_model(AUDIO_PATH)
print("yay")
print(transcription.keys())
print(transcription)
# Assign speakers
result = whisperx.assign_word_speakers(
    aligned["word_segments"],
    diarize_segments,
    pd.DataFrame(transcription["segments"])
)

# Output final transcript
for segment in result:
    print(f"{segment['start']:.1f}s - {segment['end']:.1f}s: {segment['speaker']}: {segment['text']}")
