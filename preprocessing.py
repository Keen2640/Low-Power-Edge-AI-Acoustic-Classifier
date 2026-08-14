import os
import numpy as np
import soundfile as sf
from scipy import signal

def preprocess_audio(input_path, output_path, target_sr=16000, target_duration=1.0):
    """
    Normalizes, resamples, and pads/trims a single audio file.
    """
    # 1. Load Audio File
    data, orig_sr = sf.read(input_path)
    data = data.astype(np.float32)

    # Convert Multi-channel / Stereo to Mono
    if data.ndim > 1:
        data = data[:, 0]

    # 2. Resample to Target Sample Rate
    if orig_sr != target_sr:
        num_target_samples = int(len(data) * (target_sr / orig_sr))
        data = signal.resample(data, num_target_samples)

    # 3. Standardize Duration (Pad or Trim)
    target_num_samples = int(target_sr * target_duration)
    current_samples = len(data)

    if current_samples > target_num_samples:
        # Audio is too long: Trim from the center
        start = (current_samples - target_num_samples) // 2
        data = data[start : start + target_num_samples]
    elif current_samples < target_num_samples:
        # Audio is too short: Zero-pad symmetrically (equal padding on left & right)
        pad_total = target_num_samples - current_samples
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        data = np.pad(data, (pad_left, pad_right), mode='constant')

    # 4. Peak Normalization
    # Scale max amplitude peak to 1.0 (with a small epsilon to prevent div-by-zero on silence)
    max_val = np.max(np.abs(data))
    if max_val > 1e-6:
        data = data / max_val

    # 5. Save Clean Normalized File
    sf.write(output_path, data, target_sr, subtype='PCM_16')

def process_dataset(input_dir, output_dir, target_sr=16000, target_duration=1.0):
    """
    Loops through all subfolders/files and processes the entire dataset.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Get all WAV files
    wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
    print(f"Found {len(wav_files)} files in '{input_dir}'.")
    print(f"Target Parameters: {target_sr} Hz | {target_duration}s duration | Peak Normalized\n")

    for idx, filename in enumerate(wav_files, start=1):
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)

        try:
            preprocess_audio(in_path, out_path, target_sr, target_duration)
            print(f"[{idx}/{len(wav_files)}] Processed: {filename}")
        except Exception as e:
            print(f"[{idx}/{len(wav_files)}] Failed to process {filename}: {e}")

    print(f"\nProcessing Complete! All uniform audio saved to: '{output_dir}'")

# --- EXECUTION ---
# Set your raw folder name and output folder name
RAW_FOLDER = "sounds"
CLEAN_FOLDER = "processed_sounds"

# Target: 16000 Hz, 1.0 second per clip
process_dataset(RAW_FOLDER, CLEAN_FOLDER, target_sr=16000, target_duration=1.0)
