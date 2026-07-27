import os
import csv
import time
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from collections import deque

CLASS_NAMES = ["yes", "no", "up"]

# ---- 1. Load quantized model ----
interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale, input_zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']

# ---- 2. Same spectrogram pipeline as train_tiny_model.py ----
def create_spectrogram(file_path):
    audio, sample_rate = librosa.load(file_path, sr=16000, duration=1.0)
    if len(audio) < 16000:
        audio = np.pad(audio, (0, 16000 - len(audio)), 'constant')
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=32)
    return librosa.power_to_db(spectrogram)

def make_input(file_path):
    spec = create_spectrogram(file_path)
    spec = tf.image.resize(spec[..., np.newaxis], [32, 32]).numpy()
    spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec) + 1e-10)  # same normalization as training
    # Quantize float -> int8 using the model's own scale/zero_point
    spec_q = (spec / input_scale + input_zero_point).astype(np.int8)
    return np.expand_dims(spec_q, axis=0)  # add batch dim

def predict(input_tensor):
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_details[0]['index'])[0]
    # De-quantize output back to float probabilities
    return (raw_output.astype(np.float32) - output_zero_point) * output_scale

# ---- 3. Run over real-world test clips ----
# Expected structure: real_world_test/yes/*.wav, real_world_test/no/*.wav, real_world_test/up/*.wav
TEST_DIR = "real_world_test"
results = []

for class_name in CLASS_NAMES:
    folder = os.path.join(TEST_DIR, class_name)
    if not os.path.isdir(folder):
        print(f"Warning: missing folder {folder}, skipping")
        continue
    for fname in os.listdir(folder):
        if not fname.endswith(".wav"):
            continue
        path = os.path.join(folder, fname)
        input_tensor = make_input(path)
        pred = predict(input_tensor)
        pred_label = int(np.argmax(pred))
        confidence = float(np.max(pred))
        true_label = CLASS_NAMES.index(class_name)
        results.append((fname, true_label, pred_label, confidence))

with open("real_world_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "true_label", "pred_label", "confidence"])
    writer.writerows(results)

print(f"Logged {len(results)} predictions to real_world_results.csv")

# ---- 4. Temporal smoothing (reference logic for Person A's firmware) ----
window = deque(maxlen=5)

def smoothed_prediction(raw_pred_class):
    window.append(raw_pred_class)
    most_common = max(set(window), key=window.count)
    return most_common if window.count(most_common) >= 3 else None

# ---- 5. Confidence histogram ----
if results:
    correct_conf = [c for _, t, p, c in results if t == p]
    wrong_conf = [c for _, t, p, c in results if t != p]

    plt.figure()
    plt.hist(correct_conf, bins=20, alpha=0.5, label='Correct')
    plt.hist(wrong_conf, bins=20, alpha=0.5, label='Wrong')
    plt.legend()
    plt.xlabel("Confidence")
    plt.title("Prediction Confidence: Correct vs Wrong")
    plt.savefig("confidence_histogram.png")
    print("Saved confidence_histogram.png")

# ---- 6. Confusion matrix + classification report ----
if results:
    y_true = [t for _, t, p, c in results]
    y_pred = [p for _, t, p, c in results]

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — Real World Test Set")
    plt.savefig("confusion_matrix.png")
    print("Saved confusion_matrix.png")

# ---- 7. Latency benchmark ----
if results:
    sample_folder = os.path.join(TEST_DIR, CLASS_NAMES[0])
    sample_files = [f for f in os.listdir(sample_folder) if f.endswith(".wav")]
    if sample_files:
        input_tensor = make_input(os.path.join(sample_folder, sample_files[0]))
        start = time.time()
        predict(input_tensor)
        inference_time_ms = (time.time() - start) * 1000
        print(f"Inference latency: {inference_time_ms:.2f} ms")
