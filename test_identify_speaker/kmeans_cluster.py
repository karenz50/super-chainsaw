import os
from pathlib import Path
import numpy as np
import torchaudio
import torch
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import defaultdict

# config
training_dir = Path("data/audio")
test_audio = Path("data/new.wav")
WIN_SEC = 1.5
HOP_SEC = 0.75
N_CLUSTERS = 3#len(list(training_dir.glob("*.wav"))) + 1
THRESHOLD = 0.75

# load model
encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

def windows(signal, sr, win_sec, hop_sec):
    win_len = int(win_sec * sr)
    hop_len = int(hop_sec * sr)
    for start in range(0, signal.shape[1] - win_len + 1, hop_len):
        end = start + win_len
        yield start / sr, end / sr, signal[:, start:end]

# extract multiple embeddings per clip
embeddings, clip_ids = [], []

for wav_path in training_dir.glob("*.wav"):
    signal, sr = torchaudio.load(wav_path)
    if sr != 16000:
        signal = torchaudio.functional.resample(signal, sr, 16000)
        sr = 16000
    for start, end, chunk in windows(signal, sr, WIN_SEC, HOP_SEC):
        with torch.no_grad():
            emb = encoder.encode_batch(chunk).squeeze().cpu().numpy()
            embeddings.append(emb)
            clip_ids.append(wav_path.name)

embeddings = np.stack(embeddings)

# clustering
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0)
labels = kmeans.fit_predict(embeddings)

cluster_map = defaultdict(set)

for clip_name, label in zip(clip_ids, labels):
    cluster_map[clip_name].add(label)

# print how many unique clusters per training file
print("\nclusters per training file:")
for clip_name, label_set in cluster_map.items():
    print(f"{clip_name}: {len(label_set)} unique clusters -> {sorted(label_set)}")


# pick dominant cluster
(unique, counts) = np.unique(labels, return_counts=True)
dominant = unique[np.argmax(counts)]
centroid = kmeans.cluster_centers_[dominant]

# test embedding using cosine similarity
signal, sr = torchaudio.load(test_audio)
if sr != 16000:
    signal = torchaudio.functional.resample(signal, sr, 16000)
with torch.no_grad():
    test_emb = encoder.encode_batch(signal).squeeze().cpu().numpy()

# similarity = cosine_similarity([centroid], [test_emb])[0][0]
# match = similarity >= THRESHOLD

# test embedding using k-nearest neighbors voting
k = 7
sims = cosine_similarity(embeddings, test_emb.reshape(1, -1)).flatten()
topk_indices = np.argsort(sims)[-k:]
topk_labels = labels[topk_indices]
votes = np.sum(topk_labels == dominant)

# results
# print(f"\ncosine similarity to dominant voice: {similarity:.4f}")
# print(f"threshold: {THRESHOLD}")
# print(f"same speaker? {'yes' if match else 'no'}\n")

print(f"{votes}/{k} nearest neighbors from dominant speaker")
print(f"same speaker? {'yes' if votes >= (k // 2 + 1) else 'no'}\n")

# cluster size printout
print("Cluster sizes:")
for u, c in zip(unique, counts):
    print(f"Cluster {u}: {c} segments {'(dominant)' if u == dominant else ''}")

# PCA plot
pca = PCA(n_components=2)
train_2d = pca.fit_transform(embeddings)
cent_2d  = pca.transform(centroid.reshape(1, -1))
test_2d  = pca.transform(test_emb.reshape(1, -1))

plt.figure(figsize=(8, 6))
plt.scatter(train_2d[:, 0], train_2d[:, 1], c=labels, alpha=0.7, label="audio segments")
plt.scatter(cent_2d[0, 0], cent_2d[0, 1], marker="X", s=200, color="red", label="dominant voice")
plt.scatter(test_2d[0, 0], test_2d[0, 1], marker="*", s=250, color="red", label="test clip")
plt.title("Speaker Embedding Clusters (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.show()
