import os
import torch
import whisper
from pydub import AudioSegment

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = whisper.load_model("medium", device=device)

ROOT_DIR = r"D:\voice\voiceYPL"
OUTPUT_DIR = r"D:\voice\cutting"

for group in ["Control", "PD"]:
    in_dir = os.path.join(ROOT_DIR, group)
    out_dir = os.path.join(OUTPUT_DIR, group)
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(in_dir):
        if not fname.lower().endswith(".wav"):
            continue
        fpath = os.path.join(in_dir, fname)
        print(f"Processing: {fpath}")

        # 1. Transcribe with Whisper
        result = model.transcribe(fpath, language='th', word_timestamps=True)
        
        # 2. หา timestamp ของเสียง "ย"
        start_sec = None
        for seg in result['segments']:
            if 'ย' in seg['text']:
                start_sec = seg['start']
                break
        if start_sec is None:
            print(f"Not found 'ย' in {fname}")
            continue

        # 3. ตัดเสียง 1 วินาทีหลังเจอ "ย"
        audio = AudioSegment.from_wav(fpath)
        start_ms = int(start_sec * 1000)
        end_ms = start_ms + 3000  # 3 วินาที
        segment = audio[start_ms:end_ms]
        save_path = os.path.join(out_dir, fname)
        segment.export(save_path, format="wav")
        print(f"Saved: {save_path}")
