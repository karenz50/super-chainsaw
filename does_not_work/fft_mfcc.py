import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from test_accuracy import run_test
import pandas as pd

AUDIO_PATH = "data/Anthony_Filamor_plix5ba068ee-shift-20250618_144548.wav" 
frame_duration = .5
n_clusters = 3 # number of speakers

y, sr = librosa.load(AUDIO_PATH, sr=None)
duration = librosa.get_duration(y=y, sr=sr)
print(f"Loaded {AUDIO_PATH} ({duration:.2f}s, {sr}Hz)")

frame_length = int(frame_duration * sr)
n_frames = len(y) // frame_length

segments = []
mfcc_features = []

for i in range(n_frames):
    start = i * frame_length
    end = start + frame_length
    segment = y[start:end]

    # skip empty or short segments
    if len(segment) < frame_length:
        continue

    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    
    segments.append((start / sr, end / sr))
    mfcc_features.append(mfcc_mean)

# normalize and cluster
X = StandardScaler().fit_transform(mfcc_features)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
labels = kmeans.labels_

plot_data = [
    {
        "start": float(start),
        "end": float(end),
        "speaker": f"SPEAKER_{label:02d}",
        "text": "" 
    }
    for (start, end), label in zip(segments, labels)
]

run_test(plot_data)

# print results
# print("\nSpeaker Segments")
# for (start, end), label in zip(segments, labels):
#     print(f"{start:.2f}s - {end:.2f}s: speaker {label}")

# plt.figure(figsize=(12, 4))
# librosa.display.waveshow(y, sr=sr)
# for (start, end), label in zip(segments, labels):
#     plt.axvspan(start, end, alpha=0.3, color=f"C{label}")
# plt.title("Speaker Segments")
# plt.xlabel("Time (s)")
# plt.tight_layout()
# plt.show()

# ================================ TEST ================================
# # load audio
# y, sr = librosa.load(AUDIO_PATH, sr=None)  # y is audio time series, sr is sampling rate

# fft = np.fft.fft(y)
# magnitude = np.abs(fft)
# frequency = np.linspace(0, sr, len(magnitude))

# # plot FFT
# plt.figure(figsize=(12, 4))
# plt.plot(frequency[:len(frequency)//2], magnitude[:len(magnitude)//2])
# plt.title("FFT - Frequency Spectrum")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude")
# plt.grid(True)
# plt.show()

# # mfcc for speaker and audio features
# mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# # plot MFCC
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(mfccs, x_axis='time')
# plt.colorbar(label='MFCC Coefficients')
# plt.title('MFCC')
# plt.tight_layout()
# plt.show()
