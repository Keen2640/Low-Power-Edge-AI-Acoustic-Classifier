#include <driver/i2s.h>
#include "dsps_fft2r.h"
#include "dsps_wind_hann.h"
#include "model.h"
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ============================================================================
// SHARED CONSTANTS
// ============================================================================
#define SAMPLE_RATE       16000
#define N_SAMPLES         512
#define FREQ_BINS         (N_SAMPLES / 2)
#define NUM_TIME_STEPS    32
#define MODEL_FREQ_BINS   32
#define N_MELS            MODEL_FREQ_BINS

#define MEL_FMIN          0.0f
#define MEL_FMAX          (SAMPLE_RATE / 2.0f)

// ============================================================================
// PIN CONFIGURATION & THRESHOLDS
// ============================================================================
#define FILTERED_PIN      35
#define LED_PIN           2
#define I2S_ADC_CHANNEL   ADC1_CHANNEL_7

#define CONFIDENCE_THRESHOLD 0.75f
#define SILENCE_THRESHOLD    0.05f  // Adjusted for normalized [-1.0, 1.0] samples

// Audio amplitude normalization factor (12-bit ADC centered at 2048)
#define ADC_NORMALIZE_SCALE 2048.0f

// Spectrogram Min/Max dB calibration (matching training pipeline)
#define TRAIN_MIN_DB   -88.76f
#define TRAIN_MAX_DB    34.13f

// ============================================================================
// MODEL QUANTIZATION PARAMS
// ============================================================================
#define INPUT_SCALE       0.0038099924568086863f
#define INPUT_ZERO_POINT  (-128)
#define OUTPUT_SCALE      0.00390625f
#define OUTPUT_ZERO_POINT (-128)
#define NUM_CLASSES       3

const char* CLASS_LABELS[NUM_CLASSES] = { "yes", "no", "up" };

// ============================================================================
// GLOBALS
// ============================================================================
int16_t audioBuffer[N_SAMPLES];

__attribute__((aligned(16))) float fft_input[N_SAMPLES * 2];
__attribute__((aligned(16))) float window[N_SAMPLES];
__attribute__((aligned(16))) float power_spec[FREQ_BINS];

static int   melBinPoints[N_MELS + 2];
static float melEnorm[N_MELS];

float spectrogram_matrix[NUM_TIME_STEPS][MODEL_FREQ_BINS];
int8_t quantized_input[NUM_TIME_STEPS * MODEL_FREQ_BINS];

constexpr int kTensorArenaSize = 60 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* tfl_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

// ============================================================================
// I2S SETUP
// ============================================================================
void setupI2S() {
  i2s_config_t i2sConfig = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = N_SAMPLES,
    .use_apll = false,
  };

  i2s_driver_install(I2S_NUM_0, &i2sConfig, 0, NULL);
  i2s_set_adc_mode(ADC_UNIT_1, I2S_ADC_CHANNEL);
  i2s_adc_enable(I2S_NUM_0);
}

void captureAudioBuffer() {
  size_t bytesRead = 0;
  i2s_read(I2S_NUM_0, audioBuffer, sizeof(audioBuffer), &bytesRead, portMAX_DELAY);
}

// ============================================================================
// MEL FILTERBANK SETUP
// ============================================================================
float hzToMel(float hz) {
  return 2595.0f * log10f(1.0f + hz / 700.0f);
}

float melToHz(float mel) {
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

void buildMelFilterbank() {
  float melMin = hzToMel(MEL_FMIN);
  float melMax = hzToMel(MEL_FMAX);

  float hzPoints[N_MELS + 2];
  for (int i = 0; i < N_MELS + 2; i++) {
    float mel = melMin + (melMax - melMin) * i / (float)(N_MELS + 1);
    hzPoints[i] = melToHz(mel);
    int bin = (int)floorf((N_SAMPLES + 1) * hzPoints[i] / SAMPLE_RATE);
    if (bin >= FREQ_BINS) bin = FREQ_BINS - 1;
    melBinPoints[i] = bin;
  }

  for (int m = 0; m < N_MELS; m++) {
    float leftHz  = hzPoints[m];
    float rightHz = hzPoints[m + 2];
    melEnorm[m] = 2.0f / (rightHz - leftHz);
  }
}

void setupDSP() {
  esp_err_t ret = dsps_fft2r_init_fc32(NULL, CONFIG_DSP_MAX_FFT_SIZE);
  if (ret != ESP_OK) {
    Serial.println("Error initializing ESP-DSP!");
    while (1) { delay(1000); }
  }

  dsps_wind_hann_f32(window, N_SAMPLES);
  memset(spectrogram_matrix, 0, sizeof(spectrogram_matrix));
  buildMelFilterbank();

  Serial.println("ESP32 DSP Engine Initialized.");
}

void setupModel() {
  tfl_model = tflite::GetModel(g_model);
  if (tfl_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (1) { delay(1000); }
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
    tfl_model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed!");
    while (1) { delay(1000); }
  }

  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  Serial.println("Model successfully loaded.");
}

// ============================================================================
// SIGNAL CONVERSION & DSP PROCESSING
// ============================================================================
void convertToFloatSamples(int16_t* raw, float* out, int len) {
  int32_t sum = 0;
  for (int i = 0; i < len; i++) {
    sum += (raw[i] & 0x0FFF);
  }
  float mean = (float)sum / (float)len;

  for (int i = 0; i < len; i++) {
    out[i] = ((float)(raw[i] & 0x0FFF) - mean) / ADC_NORMALIZE_SCALE;
  }
}

void compute_embedded_fft(float* raw_audio_samples) {
  for (int i = 0; i < N_SAMPLES; i++) {
    fft_input[i * 2]     = raw_audio_samples[i] * window[i];
    fft_input[i * 2 + 1] = 0.0f;
  }

  dsps_fft2r_fc32(fft_input, N_SAMPLES);
  dsps_bit_rev_fc32(fft_input, N_SAMPLES);

  for (int i = 0; i < FREQ_BINS; i++) {
    float real = fft_input[i * 2];
    float imag = fft_input[i * 2 + 1];
    power_spec[i] = real * real + imag * imag;
  }

  float mel_energies[N_MELS];
  for (int m = 0; m < N_MELS; m++) {
    int left   = melBinPoints[m];
    int center = melBinPoints[m + 1];
    int right  = melBinPoints[m + 2];

    float sum = 0.0f;
    for (int k = left; k < center; k++) {
      if (center != left) {
        float w = (float)(k - left) / (float)(center - left);
        sum += w * power_spec[k];
      }
    }
    for (int k = center; k < right; k++) {
      if (right != center) {
        float w = (float)(right - k) / (float)(right - center);
        sum += w * power_spec[k];
      }
    }
    sum *= melEnorm[m];
    mel_energies[m] = 10.0f * log10f(sum + 1e-10f);
  }

  // Shift old frames left along time axis
  for (int t = 0; t < NUM_TIME_STEPS - 1; t++) {
    for (int f = 0; f < MODEL_FREQ_BINS; f++) {
      spectrogram_matrix[t][f] = spectrogram_matrix[t + 1][f];
    }
  }

  // Insert newest frame at end
  for (int f = 0; f < MODEL_FREQ_BINS; f++) {
    spectrogram_matrix[NUM_TIME_STEPS - 1][f] = mel_energies[f];
  }
}

void applyTopDbClip() {
  float clipMax = spectrogram_matrix[0][0];
  for (int t = 0; t < NUM_TIME_STEPS; t++) {
    for (int f = 0; f < MODEL_FREQ_BINS; f++) {
      if (spectrogram_matrix[t][f] > clipMax) clipMax = spectrogram_matrix[t][f];
    }
  }

  float floorVal = clipMax - 80.0f;
  for (int t = 0; t < NUM_TIME_STEPS; t++) {
    for (int f = 0; f < MODEL_FREQ_BINS; f++) {
      if (spectrogram_matrix[t][f] < floorVal) spectrogram_matrix[t][f] = floorVal;
    }
  }
}

void quantizeSpectrogram() {
  applyTopDbClip();

  float range = TRAIN_MAX_DB - TRAIN_MIN_DB;
  if (range < 1e-6f) range = 1e-6f;

  int idx = 0;
  for (int t = 0; t < NUM_TIME_STEPS; t++) {
    for (int f = 0; f < MODEL_FREQ_BINS; f++) {
      float normalized = (spectrogram_matrix[t][f] - TRAIN_MIN_DB) / range;
      if (normalized < 0.0f) normalized = 0.0f;
      if (normalized > 1.0f) normalized = 1.0f;

      int32_t q = (int32_t)roundf(normalized / INPUT_SCALE) + INPUT_ZERO_POINT;
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      quantized_input[idx++] = (int8_t)q;
    }
  }
}

// ============================================================================
// INFERENCE & MAIN LOOP
// ============================================================================
void runInference(int8_t* input_tensor, int* out_class, float* out_confidence) {
  memcpy(model_input->data.int8, input_tensor, NUM_TIME_STEPS * MODEL_FREQ_BINS);

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    *out_class = -1;
    *out_confidence = 0.0f;
    return;
  }

  int best_class = 0;
  float best_score = -1e9f;
  for (int i = 0; i < NUM_CLASSES; i++) {
    int8_t raw_score = model_output->data.int8[i];
    float real_score = (raw_score - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
    if (real_score > best_score) {
      best_score = real_score;
      best_class = i;
    }
  }

  *out_class = best_class;
  *out_confidence = best_score;
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  setupI2S();
  setupDSP();
  setupModel();
}

void loop() {
  captureAudioBuffer();

  float floatSamples[N_SAMPLES];
  convertToFloatSamples(audioBuffer, floatSamples, N_SAMPLES);

  float minVal = floatSamples[0], maxVal = floatSamples[0];
  for (int i = 1; i < N_SAMPLES; i++) {
    if (floatSamples[i] < minVal) minVal = floatSamples[i];
    if (floatSamples[i] > maxVal) maxVal = floatSamples[i];
  }
  float energySpread = maxVal - minVal;

  // Gate out low-energy ambient noise
  if (energySpread < SILENCE_THRESHOLD) {
    digitalWrite(LED_PIN, LOW);
    return;
  }

  compute_embedded_fft(floatSamples);
  quantizeSpectrogram();

  int predicted_class = -1;
  float confidence = 0.0f;
  runInference(quantized_input, &predicted_class, &confidence);

  if (predicted_class >= 0) {
    Serial.print("yes=");
    Serial.print((model_output->data.int8[0] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE, 3);
    Serial.print(" no=");
    Serial.print((model_output->data.int8[1] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE, 3);
    Serial.print(" up=");
    Serial.print((model_output->data.int8[2] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE, 3);
    Serial.print("  -> ");
    Serial.print(CLASS_LABELS[predicted_class]);
    Serial.print(" (");
    Serial.print(confidence, 3);
    Serial.println(")");
  }

  if (predicted_class >= 0 && confidence > CONFIDENCE_THRESHOLD) {
    digitalWrite(LED_PIN, HIGH);
  } else {
    digitalWrite(LED_PIN, LOW);
  }
}