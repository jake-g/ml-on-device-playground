// Codegen: runs the classroom-exported TFLite model on the XIAO
// nRF52840 Sense. Serial protocol (115200 baud, one line per message):
//   IMU:ax,ay,az            every sample (100 Hz) in live mode
//   PROB:p0,p1,...,pN,      after each inference window
//   LAT:<us>                inference wall time
//   ARENA:<bytes>           arena bytes actually used, printed once
//   INFO:...                startup facts
//   UNSURE:<maxprob>        low-confidence window (candidate for labelling)
// Host commands:
//   REPLAY <rows>           firmware answers READY, then expects <rows> lines
//                           "ax,ay,az", acknowledging each with "." (flow
//                           control for the small CDC buffer); then runs one
//                           inference and prints PROB:/LAT: (no IMU:)
//   LIVE                    return to live sliding-window mode
//   STATUS                  reprint INFO lines
#include <LSM6DS3.h>
#include <Wire.h>

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <cmath>
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_utils.h"

#include "model_data.h"

namespace {

constexpr float kSampleRateHz = 100.0f;
constexpr unsigned long kSampleIntervalMs = (unsigned long)(1000.0f / kSampleRateHz);
constexpr int kOverlapPercent = 50;
constexpr float kUnsureThreshold = 0.60f;
constexpr int kNumFeatures = WINDOW_ROWS * NUM_AXES;
constexpr int kTensorArenaSize = 48 * 1024;

LSM6DS3 imu(I2C_MODE, 0x6A);

namespace elementwise_int8 {

struct QuantParams {
  float in_scale;
  int in_zp;
  float out_scale;
  int out_zp;
};

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  tflite::MicroContext* mc = tflite::GetMicroContext(context);
  TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);
  TfLiteTensor* in = mc->AllocateTempInputTensor(node, 0);
  TfLiteTensor* out = mc->AllocateTempOutputTensor(node, 0);
  TF_LITE_ENSURE(context, in != nullptr && out != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, in->type, out->type);
  if (in->type != kTfLiteFloat32 && in->type != kTfLiteInt8) {
    mc->DeallocateTempTfLiteTensor(in);
    mc->DeallocateTempTfLiteTensor(out);
    return kTfLiteError;
  }
  if (in->type == kTfLiteInt8) {
    auto* qp = static_cast<QuantParams*>(
        context->AllocatePersistentBuffer(context, sizeof(QuantParams)));
    if (qp == nullptr) {
      mc->DeallocateTempTfLiteTensor(in);
      mc->DeallocateTempTfLiteTensor(out);
      return kTfLiteError;
    }
    qp->in_scale = in->params.scale;
    qp->in_zp = in->params.zero_point;
    qp->out_scale = out->params.scale;
    qp->out_zp = out->params.zero_point;
    node->user_data = qp;
  }
  mc->DeallocateTempTfLiteTensor(in);
  mc->DeallocateTempTfLiteTensor(out);
  return kTfLiteOk;
}

template <float (*Fn)(float)>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* in = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* out = tflite::micro::GetEvalOutput(context, node, 0);
  const size_t n = tflite::ElementCount(*in->dims);
  if (in->type == kTfLiteFloat32) {
    const float* src = tflite::micro::GetTensorData<float>(in);
    float* dst = tflite::micro::GetTensorData<float>(out);
    for (size_t i = 0; i < n; ++i) dst[i] = Fn(src[i]);
    return kTfLiteOk;
  }
  if (in->type == kTfLiteInt8 && node->user_data != nullptr) {
    const auto* qp = static_cast<const QuantParams*>(node->user_data);
    const int8_t* src = tflite::micro::GetTensorData<int8_t>(in);
    int8_t* dst = tflite::micro::GetTensorData<int8_t>(out);
    for (size_t i = 0; i < n; ++i) {
      float x = (static_cast<int>(src[i]) - qp->in_zp) * qp->in_scale;
      float y = Fn(x);
      int q = static_cast<int>(lroundf(y / qp->out_scale)) + qp->out_zp;
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      dst[i] = static_cast<int8_t>(q);
    }
    return kTfLiteOk;
  }
  return kTfLiteError;
}

inline float OpAbs(float x) { return fabsf(x); }
inline float OpSqrt(float x) { return x > 0.0f ? sqrtf(x) : 0.0f; }
inline float OpSquare(float x) { return x * x; }

const TfLiteRegistration kRegAbs = {
    nullptr, nullptr, Prepare, Eval<OpAbs>, nullptr, tflite::BuiltinOperator_ABS, nullptr, 0};
const TfLiteRegistration kRegSqrt = {
    nullptr, nullptr, Prepare, Eval<OpSqrt>, nullptr, tflite::BuiltinOperator_SQRT, nullptr, 0};
const TfLiteRegistration kRegSquare = {
    nullptr, nullptr, Prepare, Eval<OpSquare>, nullptr, tflite::BuiltinOperator_SQUARE, nullptr, 0};

}  // namespace elementwise_int8

class GestureOpResolver : public tflite::MicroMutableOpResolver<40> {
 public:
  const TfLiteRegistration* FindOp(tflite::BuiltinOperator op) const override {
    const TfLiteRegistration* base =
        tflite::MicroMutableOpResolver<40>::FindOp(op);
    if (base == nullptr) return nullptr;
    if (op == tflite::BuiltinOperator_ABS) return &elementwise_int8::kRegAbs;
    if (op == tflite::BuiltinOperator_SQRT) return &elementwise_int8::kRegSqrt;
    if (op == tflite::BuiltinOperator_SQUARE) return &elementwise_int8::kRegSquare;
    return base;
  }
};

tflite::MicroErrorReporter error_reporter;
GestureOpResolver resolver;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

float window[kNumFeatures];
int rows_filled = 0;
int sliding_step = WINDOW_ROWS;
unsigned long last_sample_ms = 0;
bool live_mode = true;
unsigned long inference_count = 0;

// Keep in sync with deploy.ALLOWED_OPS in classroom/micro_gesture.
void RegisterOps() {
  resolver.AddFullyConnected();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddMaxPool2D();
  resolver.AddAveragePool2D();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddMean();
  resolver.AddAdd();
  resolver.AddSub();
  resolver.AddMul();
  resolver.AddRelu();
  resolver.AddRelu6();
  resolver.AddLogistic();
  resolver.AddTanh();
  resolver.AddPad();
  resolver.AddExpandDims();
  resolver.AddSqueeze();
  resolver.AddConcatenation();
  resolver.AddStridedSlice();
  resolver.AddL2Normalization();
  resolver.AddLeakyRelu();
  resolver.AddHardSwish();
  resolver.AddAbs();
  resolver.AddSqrt();
  resolver.AddSquare();
  resolver.AddMaximum();
  resolver.AddMinimum();
  resolver.AddReduceMax();
  resolver.AddArgMax();
  resolver.AddTranspose();
  resolver.AddShape();
  resolver.AddSplit();
  resolver.AddGather();
  resolver.AddSlice();
  resolver.AddCast();
  resolver.AddPack();
  resolver.AddUnpack();
}

void PrintInfo() {
  Serial.print("INFO:model_bytes=");
  Serial.print(g_model_len);
  Serial.print(",window_rows=");
  Serial.print(WINDOW_ROWS);
  Serial.print(",classes=");
  for (int i = 0; i < NUM_CLASSES; i++) {
    Serial.print(GESTURES[i]);
    if (i + 1 < NUM_CLASSES) Serial.print("|");
  }
  Serial.print(",input_type=");
  Serial.print(input->type == kTfLiteInt8 ? "int8" : "float32");
  Serial.print(",output_type=");
  Serial.print(output->type == kTfLiteInt8 ? "int8" : "float32");
  Serial.print(",step=");
  Serial.println(sliding_step);
  Serial.print("ARENA:");
  Serial.println(interpreter->arena_used_bytes());
}

void FillInput() {
  if (input->type == kTfLiteInt8) {
    const float scale = input->params.scale;
    const int zp = input->params.zero_point;
    for (int i = 0; i < kNumFeatures; i++) {
      int q = (int)lroundf(window[i] / scale) + zp;
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      input->data.int8[i] = (int8_t)q;
    }
  } else {
    for (int i = 0; i < kNumFeatures; i++) input->data.f[i] = window[i];
  }
}

float OutputProb(int i) {
  if (output->type == kTfLiteInt8) {
    return (output->data.int8[i] - output->params.zero_point) *
           output->params.scale;
  }
  return output->data.f[i];
}

void RunInference() {
  FillInput();
  unsigned long t0 = micros();
  TfLiteStatus status = interpreter->Invoke();
  unsigned long dt = micros() - t0;
  if (status != kTfLiteOk) {
    Serial.println("ERR:invoke failed");
    return;
  }
  inference_count++;
  float max_p = 0.0f;
  Serial.print("PROB:");
  for (int i = 0; i < NUM_CLASSES; i++) {
    float p = OutputProb(i);
    if (p > max_p) max_p = p;
    Serial.print(p, 6);
    Serial.print(",");
  }
  Serial.print("\n");
  Serial.print("LAT:");
  Serial.println(dt);
  if (max_p < kUnsureThreshold) {
    Serial.print("UNSURE:");
    Serial.println(max_p, 3);
  }
}

void SlideWindow() {
  if (sliding_step < WINDOW_ROWS) {
    const int keep = (WINDOW_ROWS - sliding_step) * NUM_AXES;
    memmove(window, window + sliding_step * NUM_AXES, keep * sizeof(float));
    rows_filled = WINDOW_ROWS - sliding_step;
  } else {
    rows_filled = 0;
  }
}

bool ReadLine(String& out, unsigned long timeout_ms) {
  unsigned long start = millis();
  out = "";
  while (millis() - start < timeout_ms) {
    while (Serial.available()) {
      char c = (char)Serial.read();
      if (c == '\n') {
        out.trim();
        return true;
      }
      if (c != '\r') out += c;
    }
  }
  return false;
}

void HandleReplay(int rows) {
  if (rows != WINDOW_ROWS) {
    Serial.print("ERR:replay rows must be ");
    Serial.println(WINDOW_ROWS);
    return;
  }
  String line;
  Serial.println("READY");
  for (int r = 0; r < rows; r++) {
    if (!ReadLine(line, 2000)) {
      Serial.println("ERR:replay timeout");
      return;
    }
    float v[NUM_AXES];
    int idx = 0;
    int from = 0;
    for (int a = 0; a < NUM_AXES; a++) {
      int comma = line.indexOf(',', from);
      String tok = comma < 0 ? line.substring(from) : line.substring(from, comma);
      v[idx++] = tok.toFloat();
      from = comma + 1;
    }
    for (int a = 0; a < NUM_AXES; a++) window[r * NUM_AXES + a] = v[a];
    Serial.println(".");
  }
  RunInference();
  rows_filled = 0;
}

void HandleCommand(const String& cmd) {
  if (cmd.startsWith("REPLAY")) {
    live_mode = false;
    int rows = cmd.substring(6).toInt();
    HandleReplay(rows);
  } else if (cmd == "LIVE") {
    live_mode = true;
    rows_filled = 0;
    Serial.println("INFO:live");
  } else if (cmd == "STATUS") {
    PrintInfo();
  }
}

}  // namespace

void setup() {
  Serial.begin(115200);
  unsigned long start = millis();
  while (!Serial && millis() - start < 4000) {
  }

  sliding_step = WINDOW_ROWS * (100 - kOverlapPercent) / 100;
  if (sliding_step < 1) sliding_step = 1;

  if (imu.begin() != 0) {
    Serial.println("ERR:imu init failed");
  }

  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERR:model schema mismatch");
    while (true) delay(1000);
  }
  RegisterOps();
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERR:AllocateTensors failed (arena too small or op missing)");
    while (true) delay(1000);
  }
  input = interpreter->input(0);
  output = interpreter->output(0);
  int in_elems = 1;
  for (int i = 0; i < input->dims->size; i++) in_elems *= input->dims->data[i];
  if (in_elems != kNumFeatures) {
    Serial.print("ERR:input elems ");
    Serial.print(in_elems);
    Serial.print(" != ");
    Serial.println(kNumFeatures);
    while (true) delay(1000);
  }
  PrintInfo();
}

void loop() {
  if (Serial.available()) {
    String cmd;
    if (ReadLine(cmd, 50)) HandleCommand(cmd);
  }
  if (!live_mode) return;

  unsigned long now = millis();
  if (now - last_sample_ms < kSampleIntervalMs) return;
  last_sample_ms = now;

  float ax = imu.readFloatAccelX();
  float ay = imu.readFloatAccelY();
  float az = imu.readFloatAccelZ();
  Serial.print("IMU:");
  Serial.print(ax);
  Serial.print(",");
  Serial.print(ay);
  Serial.print(",");
  Serial.print(az);
  Serial.print("\n");

  window[rows_filled * NUM_AXES + 0] = ax;
  window[rows_filled * NUM_AXES + 1] = ay;
  window[rows_filled * NUM_AXES + 2] = az;
  rows_filled++;
  if (rows_filled == WINDOW_ROWS) {
    RunInference();
    SlideWindow();
  }
}
