import argparse
from datasets import load_dataset
from torch import cosine_similarity
import whisper
import jiwer, json, os, torch, torchaudio
import numpy as np
from scipy.signal import correlate2d
from scipy.io.wavfile import write
from cfm_tts_pytorch.trainer import HFDataset, collate_fn
from pathlib import Path
from cfm_tts_pytorch import C3TTS

# -------------------------
# Argument Parser
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help='Path to E2TTS checkpoint (e.g., e2tts.pt)')
parser.add_argument('--save_path', type=str, default='./MushanWGLOBEEval', help='Path to save outputs')
args = parser.parse_args()

# -------------------------
# Model Setup
# -------------------------
e2tts = C3TTS(
    cond_drop_prob = 0.25,
    transformer = dict(
        dim = 1024,
        depth = 24,
        heads = 16,
        ff_mult=4
    ),
    mel_spec_kwargs = dict(
        filter_length = 1024,
        hop_length = 256,
        win_length = 1024,
        n_mel_channels = 100,
        sampling_rate = 24000,
    )
)

checkpoint = torch.load(args.model_path, map_location='cpu')
if 'model_state_dict' in checkpoint:
    e2tts.load_state_dict(checkpoint['model_state_dict'])
else:
    e2tts.load_state_dict(checkpoint['ema_model_state_dict'])

# -------------------------
# Setup Paths and Data
# -------------------------
os.makedirs(args.save_path, exist_ok=True)
test_dataset = load_dataset("hishab/MushanWGLOBEEval", split='test[:5]')
dataset_load = HFDataset(test_dataset)

# ASR model for evaluation
asr_model = whisper.load_model("small")

# MFCC transform
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=24000,
    n_mfcc=13,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
)

wers = []
mfccs = []
spec_corrs = []

# -------------------------
# Evaluation Loop
# -------------------------
for index, (sample, sample_load) in enumerate(zip(test_dataset, dataset_load)):
    # Save ground-truth audio
    int16_array = np.int16(sample['audio']['array'] * 32767)
    gt_wav_path = f"{args.save_path}/test_dataset_{index}.wav"
    write(gt_wav_path, sample['audio']['sampling_rate'], int16_array)

    # Save transcript
    with open(f"{args.save_path}/test_dataset_{index}.txt", "w") as f:
        f.write(sample['transcript'])

    audio, sr = torchaudio.load(gt_wav_path)

    with torch.inference_mode():
        duration = sample_load['mel_spec'].shape[1] * 2
        generated = e2tts.sample(
            cond = audio,
            text = [sample_load['text'] + ' ' + sample_load['text']],
            duration = duration,
            steps = 16,
            cfg_strength = 0.5,
            save_to_filename=f"{args.save_path}/test_dataset_{index}_generated.wav"
        )

    # Load generated audio
    audio, sr = torchaudio.load(f"{args.save_path}/test_dataset_{index}_generated.wav")
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio = resampler(audio)
    audio = audio.squeeze().numpy()

    # Whisper transcription
    transcription = asr_model.transcribe(audio, language='en')["text"].strip()

    # WER
    wer = jiwer.wer(sample['transcript'], transcription)
    wers.append(wer)

    # Uncomment below to include similarity metrics

    # mfcc = mfcc_transform(audio)
    # mfcc_ref = mfcc_transform(torch.tensor(sample['audio']['array']).unsqueeze(0))
    # mfcc_sim = cosine_similarity(mfcc, mfcc_ref).mean().item()
    # mfccs.append(mfcc_sim)

    # spec1 = torchaudio.transforms.Spectrogram()(torch.tensor(audio)).squeeze().numpy()
    # spec2 = torchaudio.transforms.Spectrogram()(torch.tensor(sample['audio']['array'])).squeeze().numpy()
    # corr = correlate2d(spec1, spec2, mode='valid')
    # similarity_score = np.max(corr)
    # spec_corrs.append(similarity_score)

# -------------------------
# Save Results
# -------------------------
with open(f"{args.save_path}/test_dataset_evaluation.json", "w") as f:
    json.dump({
        "wer": wers,
        "mfcc": mfccs,
        "spec_corr": spec_corrs
    }, f)

print("✅ Evaluation complete. Results saved to:", args.save_path)
