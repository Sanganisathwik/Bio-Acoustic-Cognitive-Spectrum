import os
import random
import shutil
import librosa
import soundfile as sf
import numpy as np

# Updated paths
BASE_RAW_DIR = r"C:\Users\sanga\Downloads\SATHWIK\Documents\src\data\raw\dataset-11"
OUT_DIR     = r"C:\Users\sanga\Downloads\SATHWIK\Documents\src\data"
TARGET_SR   = 16000

# Sample counts
N_BIO    = 3500
N_NONBIO = 1500

os.makedirs(os.path.join(OUT_DIR, "bio"),    exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "nonbio"), exist_ok=True)

# Clear existing processing if any to start fresh
for d in ["bio", "nonbio"]:
    p = os.path.join(OUT_DIR, d)
    for f in os.listdir(p):
        if f.endswith(".wav"):
            os.remove(os.path.join(p, f))

# Identify folders
folders = [d for d in os.listdir(BASE_RAW_DIR) if os.path.isdir(os.path.join(BASE_RAW_DIR, d))]
bio_folders = [f for f in folders if f.lower() != "noise"]
nonbio_folders = [f for f in folders if f.lower() == "noise"]

print(f"Bio Folders: {len(bio_folders)}")
print(f"Non-Bio Folders: {len(nonbio_folders)}")

# Process Non-Bio (Noise)
nonbio_files = []
for fld in nonbio_folders:
    pth = os.path.join(BASE_RAW_DIR, fld)
    for root, dirs, files in os.walk(pth):
        nonbio_files.extend([os.path.join(root, f) for f in files if f.lower().endswith(('.wav', '.mp3'))])

random.shuffle(nonbio_files)
selected_nonbio = nonbio_files
N_NONBIO = len(selected_nonbio)

print(f"Processing {len(selected_nonbio)} non-bio files...")
for i, src in enumerate(selected_nonbio):
    try:
        y, sr = librosa.load(src, sr=TARGET_SR, mono=True)
        out_path = os.path.join(OUT_DIR, "nonbio", f"nonbio_{i:04d}.wav")
        sf.write(out_path, y, TARGET_SR)
    except Exception as e:
        print(f"Error processing {src}: {e}")

# Process Bio (All species)
bio_files = []
# Give each folder a portion (2000 / 10 = 200 samples/folder approximately)
samples_per_folder = N_BIO // len(bio_folders)
for fld in bio_folders:
    pth = os.path.join(BASE_RAW_DIR, fld)
    f_list = []
    for root, dirs, files in os.walk(pth):
        f_list.extend([os.path.join(root, f) for f in files if f.lower().endswith(('.wav', '.mp3'))])
    random.shuffle(f_list)
    bio_files.extend(f_list[:samples_per_folder])

# If we need more to hit N_BIO (due to integer division)
if len(bio_files) < N_BIO:
    remaining = N_BIO - len(bio_files)
    # Just draw from all
    all_bio = []
    for fld in bio_folders:
        pth = os.path.join(BASE_RAW_DIR, fld)
        for root, dirs, files in os.walk(pth):
            all_bio.extend([os.path.join(root, f) for f in files if f.lower().endswith(('.wav', '.mp3'))])
    random.shuffle(all_bio)
    # Add what we don't already have
    existing = set(bio_files)
    for f in all_bio:
        if f not in existing:
            bio_files.append(f)
            remaining -= 1
        if remaining <= 0:
            break

print(f"Processing {len(bio_files)} bio files...")
for i, src in enumerate(bio_files):
    try:
        y, sr = librosa.load(src, sr=TARGET_SR, mono=True)
        out_path = os.path.join(OUT_DIR, "bio", f"bio_{i:04d}.wav")
        sf.write(out_path, y, TARGET_SR)
    except Exception as e:
        print(f"Error processing {src}: {e}")

print(f"\nDone! Data organized in: {OUT_DIR}")
