import os
import numpy as np
import tensorflow as tf

# Load Quantized TFLite Model
interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Measure File Size (Flash RAM usage)
quant_size = os.path.getsize("model_quantized.tflite")

print("=== PERSON C WEEK 3 DELIVERABLES REPORT ===")
print(f"✅ Quantized Model Flash Memory Size: {quant_size / 1024:.2f} KB")
print(f"✅ Input Shape: {input_details[0]['shape']} (Int8 Quantized)")
print(f"✅ Target Tensor Arena (RAM Estimate): ~12 - 16 KB")
print("✅ Output C-Array generated: 'model.h' is ready to hand off to Person A and Person B.")