# 🎙️ Low-Power Edge AI Acoustic Classifier
**Module:** TinyML & Neural Network Optimization Pipeline  
**Role:** Person C — TinyML & Deep Learning Engineer  
**Target Hardware:** ESP32 / Raspberry Pi Pico (ARM Cortex-M / Tensilica Xtensa)  

---

## 📌 Project Overview
This repository contains the Machine Learning and Model Optimization pipeline for an **Embedded Edge-AI Acoustic Classifier**. The primary goal of this phase is to design, train, quantize, and export a compact Convolutional Neural Network (CNN) capable of classifying audio spectrograms in real-time under extreme hardware constraints ($<32\text{ KB}$ Flash, $<16\text{ KB}$ Tensor Arena RAM).

---

## 📈 Visualizations & Performance Analysis

### 1. Audio Preprocessing Pipeline (Time Domain to Spectrogram)
Raw 1D audio waveforms are transformed into 2D frequency representations using Short-Time Fourier Transform (STFT) / Mel-Spectrogram features before being fed into the quantized neural network.

<p align="center">
  <img src="assets/waveform_spectrogram.png" width="85%" alt="Audio Waveform and Spectrogram Analysis">
</p>

---

### 2. Training Convergence & Accuracy Metrics
The compact CNN achieved a **Final Test Accuracy of 79.44%** over 25 epochs. The training curves demonstrate smooth convergence between train/validation accuracy and loss, validating effective generalization without significant overfitting.

<p align="center">
  <img src="assets/training_metrics.png" width="90%" alt="Training vs Validation Accuracy and Loss Curves">
</p>

---

## 🛠️ Key Technical Contributions (Person C)

### 1. Architectural Shrinking & Spectrogram Normalization
* Designed a lightweight 2D Convolutional Neural Network optimized for spatial-temporal audio features.
* Downsampled the input spectrogram resolution to **$32 \times 32 \times 1$**, reducing overall input tensor payload by **75%** compared to baseline $64 \times 64$ pipelines.
* Integrated on-the-fly gaussian noise data augmentation to mitigate overfitting and improve real-world analog noise tolerance.

### 2. Post-Training INT8 Quantization
* Leveraged TensorFlow Lite Converter with a **Representative Dataset Calibration Generator** to quantize all full-precision floating-point weights and activation layers (`float32`) down to signed 8-bit integers (`int8`).
* Preserved edge inference classification accuracy while achieving a **4x compression ratio**.

### 3. C-Header Array Generation (`model.h`)
* Transformed the flat buffer `.tflite` binary file into a static C byte array using Unix binary utilities (`xxd`).
* Configured target storage specifiers (`const`) to ensure the model weights are stored in the microcontroller’s **Flash Memory (ROM)** rather than consuming valuable dynamic memory (**SRAM**).

### 4. Embedded Inference Simulation
* Implemented a host-side local verification script utilizing the **TensorFlow Lite Interpreter** to validate quantized tensor shapes, quantization scale/zero-point parameters, and Tensor Arena RAM allocations.

---

## 📊 Performance Benchmarks & Deliverables

| Metric | Target Specification | Achieved Metric | Status |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | $> 75.0\%$ | **$79.44\%$** | 🟢 Optimal |
| **Model Size (Flash)** | $< 30.0\text{ KB}$ | **$15.15\text{ KB}$** | 🟢 Optimal |
| **Input Tensor Shape** | $32 \times 32 \times 1$ | `[1, 32, 32, 1]` (INT8) | 🟢 Verified |
| **Est. Tensor Arena (RAM)** | $< 20.0\text{ KB}$ | **$\sim 12\text{--}16\text{ KB}$** | 🟢 Optimal |
| **Target Firmware Header** | C Source Export | `model.h` (`g_model`) | 🟢 Exported |

---

## 📁 Repository Structure & Artifacts

```text
TinyML_PersonC/
├── assets/
│   ├── training_metrics.png        # Training vs Validation accuracy/loss graphs
│   └── waveform_spectrogram.png    # Time-domain waveform and spectrogram plot
├── mini_speech_commands/           # Downsampled multi-class audio dataset
├── train_tiny_model.py             # Complete training & INT8 quantization script
├── model_quantized.tflite          # Compressed TensorFlow Lite FlatBuffer binary
├── model.h                         # C Header Byte Array (Deployed to MCU PlatformIO src/)
└── test_micro_inference.py         # Local TFLite Micro execution test script
