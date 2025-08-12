import librosa
import numpy as np
import parselmouth
import pandas as pd
import os

def extract_features(filepath):
    feature_status = {}
    
    # 1. MFCC
    try:
        y, sr = librosa.load(filepath, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        feature_status['mfcc'] = not (np.isnan(mfcc_mean).any() or np.isnan(mfcc_std).any())
    except:
        mfcc_mean = np.array([np.nan]*13)
        mfcc_std = np.array([np.nan]*13)
        feature_status['mfcc'] = False

    # 2. Fundamental Frequency (F0)
    try:
        snd = parselmouth.Sound(filepath)
        pitch = snd.to_pitch()
        f0_values = pitch.selected_array['frequency']
        f0_values = f0_values[f0_values > 0]
        f0_mean = np.mean(f0_values) if len(f0_values) > 0 else np.nan
        f0_std = np.std(f0_values) if len(f0_values) > 0 else np.nan
        feature_status['f0'] = not (np.isnan(f0_mean) or np.isnan(f0_std))
    except:
        f0_mean = f0_std = np.nan
        feature_status['f0'] = False

    # 3. Jitter & Shimmer
    try:
        pointProcess = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter = parselmouth.praat.call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = parselmouth.praat.call([snd, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        feature_status['jitter'] = not np.isnan(jitter)
        feature_status['shimmer'] = not np.isnan(shimmer)
    except:
        jitter = shimmer = np.nan
        feature_status['jitter'] = False
        feature_status['shimmer'] = False

    # 4. HNR
    try:
        hnr = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr_mean = parselmouth.praat.call(hnr, "Get mean", 0, 0)
        feature_status['hnr'] = not np.isnan(hnr_mean)
    except:
        hnr_mean = np.nan
        feature_status['hnr'] = False

    # 5. Formants (F1, F2, F3)
    try:
        formant = snd.to_formant_burg()
        F1 = [formant.get_value_at_time(1, t) for t in np.linspace(0, snd.duration, 10)]
        F2 = [formant.get_value_at_time(2, t) for t in np.linspace(0, snd.duration, 10)]
        F3 = [formant.get_value_at_time(3, t) for t in np.linspace(0, snd.duration, 10)]
        f1_mean = np.nanmean(F1)
        f2_mean = np.nanmean(F2)
        f3_mean = np.nanmean(F3)
        feature_status['F1'] = not np.isnan(f1_mean)
        feature_status['F2'] = not np.isnan(f2_mean)
        feature_status['F3'] = not np.isnan(f3_mean)
    except:
        f1_mean = f2_mean = f3_mean = np.nan
        feature_status['F1'] = feature_status['F2'] = feature_status['F3'] = False

    # สร้าง dictionary ของฟีเจอร์ทั้งหมด
    feats = {f"mfcc{i+1}_mean": mfcc_mean[i] for i in range(len(mfcc_mean))}
    feats.update({f"mfcc{i+1}_std": mfcc_std[i] for i in range(len(mfcc_std))})
    feats.update({
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "jitter": jitter,
        "shimmer": shimmer,
        "hnr": hnr_mean,
        "F1": f1_mean, "F2": f2_mean, "F3": f3_mean
    })
    feats.update({f"{k}_ok": v for k, v in feature_status.items()})
    return feats

# 🔍 Loop ทุกไฟล์ในโฟลเดอร์
root_dir = r"D:\voice\cutting"
data = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.lower().endswith('.wav'):
            file_path = os.path.join(dirpath, filename)
            label = os.path.basename(dirpath)
            try:
                feats = extract_features(file_path)
                feats["label"] = label
                feats["filename"] = filename
                data.append(feats)
                missing = [k for k in ['mfcc','f0','jitter','shimmer','hnr','F1','F2','F3'] if not feats[k+'_ok']]
                if len(missing) == 0:
                    print(f"✅ ครบทุกฟีเจอร์: {file_path}")
                else:
                    print(f"⚠️ ขาดบางฟีเจอร์: {file_path} | {missing}")
            except Exception as e:
                print(f"❌ Error in {file_path}: {e}")

# 💾 บันทึกเป็น CSV
df = pd.DataFrame(data)
df.to_csv("voice_features.csv", index=False)
print("📁 Saved all features to voice_features.csv")
