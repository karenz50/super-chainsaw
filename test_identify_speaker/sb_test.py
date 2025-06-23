import pickle, numpy as np, torchaudio, torch
from speechbrain.pretrained import EncoderClassifier
from pathlib import Path

TEMPLATE_PATH = Path("voice_template.pkl")
encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                         savedir="sb_ecapa").eval()
template = pickle.load(TEMPLATE_PATH.open("rb"))

def cosine_sim(a, b):  # 1 = identical, -1 = opposite
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def verify_clip(clip_path, threshold=0.65):
    wav, sr = torchaudio.load(clip_path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    with torch.no_grad():
        test_emb = encoder.encode_batch(wav).squeeze().cpu().numpy()
    score = cosine_sim(template, test_emb)
    same = score >= threshold
    return score, same

score, same = verify_clip("data/audio/new.wav")
print(f"cosine similarity: {score:.3f}")
print("SAME SPEAKER!!!!" if same else "different speaker")
