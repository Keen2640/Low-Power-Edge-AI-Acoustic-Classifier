import os
import zipfile
import urllib.request
import numpy as np
import tensorflow as tf
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 1. Download & Extract Dataset directly into local workspace
data_dir = "mini_speech_commands"
zip_name = "mini_speech_commands.zip"
dataset_url = "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"

if not os.path.exists(data_dir):
    print("Downloading dataset to local folder...")
    urllib.request.urlretrieve(dataset_url, zip_name)
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Dataset ready!")

# 2. Audio Processing Helper Functions
def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def create_spectrogram(file_path, augment=False):
    audio, sample_rate = librosa.load(file_path, sr=16000, duration=1.0)
    if len(audio) < 16000:
        audio = np.pad(audio, (0, 16000 - len(audio)), 'constant')
    if augment:
        audio = add_noise(audio)
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=32)
    return librosa.power_to_db(spectrogram)

# 3. Load & Downsample Data to 32x32 Spectrograms
X, y = [], []
classes = ["yes", "no", "up"]

for label, class_name in enumerate(classes):
    folder = os.path.join(data_dir, class_name)
    files = [f for f in os.listdir(folder) if f.endswith('.wav')][:300]
    print(f"Processing class '{class_name}' ({len(files)} files)...")
    for file in files:
        path = os.path.join(folder, file)
        try:
            # Resizing to 32x32 explicitly for microcontrollers
            spec = create_spectrogram(path, augment=False)
            spec = tf.image.resize(spec[..., np.newaxis], [32, 32]).numpy()
            X.append(spec)
            y.append(label)

            spec_noisy = create_spectrogram(path, augment=True)
            spec_noisy = tf.image.resize(spec_noisy[..., np.newaxis], [32, 32]).numpy()
            X.append(spec_noisy)
            y.append(label)
        except Exception:
            pass

X = np.array(X, dtype="float32")
X = (X - np.min(X)) / (np.max(X) - np.min(X))
y_encoded = to_categorical(np.array(y))

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 4. Build Reduced CNN Architecture
model = Sequential([
    Conv2D(8, (3,3), activation='relu', input_shape=(32,32,1)),
    MaxPooling2D((2,2)),
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.25),
    Flatten(),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("\n--- Training Compact Microcontroller Model ---")
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 5. INT8 Quantization
def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(100):
        yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

with open("model_quantized.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("\n Success! Saved INT8 model as 'model_quantized.tflite'")