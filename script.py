import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

def process_audio(wav_path):
    # 1. Load the WAV file
    # sample_rate (fs) is samples per second. data is the raw amplitude array.
    sample_rate, data = wavfile.read(wav_path)
    
    # Handle stereo files (convert to mono by taking the first channel)
    if len(data.shape) > 1:
        data = data[:, 0]
        
    print(f"Processing: {os.path.basename(wav_path)}")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Total Samples: {len(data)}")
    print(f"Duration: {len(data) / sample_rate:.2f} seconds\n")

    # Time vector for the x-axis of the waveform
    duration = len(data) / sample_rate
    time_axis = np.linspace(0, duration, len(data))

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # --- Plot 1: 1D Time-Domain Waveform ---
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, data, color='darkcyan')
    plt.title(f"Time-Domain Waveform: {os.path.basename(wav_path)}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)

    # --- Plot 2: 2D Spectrogram ---
    # nperseg: The size of each FFT window (corresponds to N-point FFT)
    # noverlap: How many samples overlap between windows (50% to 75% is standard)
    nperseg = 512
    noverlap = 256
    
    frequencies, times, spec_matrix = signal.spectrogram(
        data, 
        fs=sample_rate, 
        nperseg=nperseg, 
        noverlap=noverlap
    )

    plt.subplot(2, 1, 2)
    # CRITICAL DSP STEP: Convert power to Decibels (log scale). 
    # Humans hear logarithmically, and AI models learn much better from log-scale intensity.
    # Added 1e-10 to prevent taking the log of zero.
    log_spec = 10 * np.log10(spec_matrix + 1e-10)
    
    # Plot using pcolormesh
    plt.pcolormesh(times, frequencies, log_spec, shading='gouraud', cmap='viridis')
    plt.title("Spectrogram (Frequency Domain over Time)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Intensity (dB)")
    
    # Optional: Limit the y-axis if you only care about lower frequencies 
    # (e.g., human speech is mostly under 4000 Hz)
    # plt.ylim(0, 4000) 

    plt.tight_layout()
    plt.show()

# Example usage: Drop a test .wav file in your directory and run it
# process_audio("test_clap.wav")