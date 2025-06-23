# run once to enrollment in speechbrain

import numpy as np, torchaudio, torch
from speechbrain.pretrained import EncoderClassifier
from pathlib import Path, PurePath
import pickle, glob

COMMON_CLIPS_DIR = Path("data/audio")  # contains 4 mp4/wav files
TEMPLATE_PATH    = Path("data/voice_template.pkl")  # where we’ll store centroid

encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", savedir="sb_ecapa"
).eval()

def embed_file(path):
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    with torch.no_grad():
        emb = encoder.encode_batch(wav) # (1, 192)
    return emb.squeeze().cpu().numpy()

# extract embeddings for each clip
embeddings = [embed_file(p) for p in glob.glob(str(COMMON_CLIPS_DIR / "*.*"))]

# create template = centroid
template = np.mean(np.stack(embeddings), axis=0)

# save for later
with TEMPLATE_PATH.open("wb") as f:
    pickle.dump(template, f)

print("enrollment done, template saved")
