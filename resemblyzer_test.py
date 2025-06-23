from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
from test_accuracy import run_test

# Load and preprocess audio
wav = preprocess_wav(Path("audio.wav"))
encoder = VoiceEncoder()
embed, partial_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)

# Cluster embeddings
kmeans = KMeans(n_clusters=3).fit(partial_embeds)  # Adjust clusters as needed

segments = []
base_time = datetime(1970, 1, 1)
sampling_rate = 16000

for wav_slice, label in zip(wav_splits, kmeans.labels_):
    start = wav_slice.start / sampling_rate
    end = wav_slice.stop / sampling_rate

    segments.append({
        "speaker": f"SPEAKER_{label:02}",
        "start": start,
        "end": end,
        "text": ""  # You can fill this later with transcribed text
    })

run_test(segments)