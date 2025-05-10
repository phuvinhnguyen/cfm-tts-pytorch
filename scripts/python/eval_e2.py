from datasets import load_dataset
from torch import cosine_similarity
import whisper
import jiwer, json, os, torch, torchaudio
import numpy as np
from scipy.signal import correlate2d
from scipy.io.wavfile import write
from cfm_tts_pytorch.trainer import HFDataset, collate_fn
from pathlib import Path
from cfm_tts_pytorch import E2TTS


e2tts = E2TTS(
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

checkpoint = torch.load('e2tts.pt', map_location='cpu')
if 'model_state_dict' in checkpoint:
    e2tts.load_state_dict(checkpoint['model_state_dict'])
else:
    e2tts.load_state_dict(checkpoint['ema_model_state_dict'])

# Get asr model
asr_model = whisper.load_model("small")

save_path = "./MushanWGLOBEEval"
os.makedirs(save_path, exist_ok=True)
test_dataset = load_dataset("hishab/MushanWGLOBEEval", split='test[:5]')
dataset_load = HFDataset(test_dataset)

# define mfcc
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=24000,
    n_mfcc=13,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},
)

wers = []
mfccs = []
spec_corrs = []
for index, (sample, sample_load) in enumerate(zip(test_dataset, dataset_load)):
    int16_array = np.int16(sample['audio']['array'] * 32767)
    write(f"{save_path}/test_dataset_{index}.wav", sample['audio']['sampling_rate'], int16_array)

    with open(f"{save_path}/test_dataset_{index}.txt", "w") as f:
        f.write(sample['transcript'])
    
    audio, sr = torchaudio.load(Path(f"{save_path}/test_dataset_{index}.wav").expanduser())

    with torch.inference_mode():
        duration = sample_load['mel_spec'].shape[1] * 2
        generated = e2tts.sample(
            cond = audio,
            text = [sample_load['text'] + ' ' + sample_load['text']],
            duration = duration,
            steps = 16,
            cfg_strength = 0.5,
            save_to_filename=f"{save_path}/test_dataset_{index}_generated.wav"
        )
    
    # Load and preprocess audio for Whisper
    audio, sr = torchaudio.load(f"{save_path}/1.test_dataset_{index}_generated.wav")
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio = resampler(audio)
    audio = audio.squeeze().numpy()

    # Whisper expects audio length in seconds (not normalized)
    transcription = asr_model.transcribe(audio, language='en')["text"].strip()
    
    # Compute WER
    wer = jiwer.wer(sample['transcript'], transcription)
    wers.append(wer)

    # # Compute MFCC + Cosine Similarity
    # mfcc = mfcc_transform(audio)
    # mfcc_ref = mfcc_transform(sample['audio'])
    # mfccs.append(cosine_similarity(mfcc, mfcc_ref))
    
    # # Spectrogram Cross-Correlation
    # spec1 = torchaudio.transforms.Spectrogram()(audio).squeeze().numpy()
    # spec2 = torchaudio.transforms.Spectrogram()(sample['audio']).squeeze().numpy()
    # corr = correlate2d(spec1, spec2, mode='valid')
    # similarity_score = np.max(corr)
    # spec_corrs.append(similarity_score)

    # print(wers[-1], mfccs[-1], spec_corrs[-1])

with open(f"{save_path}/test_dataset_evaluation.json", "w") as f:
    json.dump({
        "wer": wer,
        "mfcc": mfccs,
        "spec_corr": spec_corrs
    }, f)  
    