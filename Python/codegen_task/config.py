"""Configuration, prompts, baseline solution, and cross-run memory.

Sections: task constants and metric weights; the 8-section student
instructions; refinement directions; the reference baseline solution;
starting archetypes and the insights store that feeds "what previous
runs learned" into later prompts.
"""

from __future__ import annotations

import json
import os
import re
from typing import Mapping, Sequence

from codegen_classroom import scoring

from codegen_task import data_loader
from codegen_task import deploy
from codegen_task import toolkit

TASK_NAME = "micro_gesture"
TASK_TITLE = "XIAO nRF52840 Gesture Classifier (TFLite Micro)"
STUDENT_FUNCTION_NAME = "solution"

DEFAULT_STUDENT_MODELS: Sequence[str] = (
    "gemini-4-argon",
    "gemini-3.5-flash",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash-lite",
)

RUN_DEFAULTS = {
    # Number of parallel student agents in the classroom population.
    "num_students": 4,
    # Number of iterative refinement and crossover cycles per run.
    "num_cycles": 12,
    # Per-call wall-clock timeout in seconds for a single fold execution.
    "timeout_s": 150,
    # Wall-clock timeout in seconds for final evaluation on all learning data.
    "final_eval_timeout_s": 480,
    # Consecutive non-improving cycles before rolling back / mutating from best.
    "patience": 1,
    # Consecutive stalled mutations before a student is excluded and respawned.
    "max_mutations": 4,
    # Maximum LLM completion tokens per student generation call.
    "max_output_tokens": 32768,
    # Number of stratified grouped cross-validation folds scored per cycle.
    "cv_folds": 3,
    # Fraction of students initialized from the baseline seed in cycle 0.
    "seed_fraction": 0.25,
    # Start all students from the minimal v1 baseline without pre-built seeds.
    "no_seed": True,
    # Disable cross-run memory and learned-lever injection by default.
    "insights": False,
}

# Prediction quality 0.68, on-device cost 0.22, deploy hygiene 0.10.
DEFAULT_METRIC_WEIGHTS: Mapping[str, float] = {
    # Unweighted macro-averaged F1 score across all gesture classes.
    "f1_macro": 0.42,
    # Mean per-class recall across all gesture classes.
    "balanced_accuracy": 0.09,
    # Minimum single-class recall to penalize ignored or collapsed gestures.
    "worst_class_recall": 0.11,
    # False-positive trigger rate on Null windows (lower is better).
    "null_fpr": 0.06,
    # Serialized TFLite flatbuffer size in KB against the soft flash budget.
    "model_kb": 0.08,
    # Million floating-point or MAC operations per single-window inference.
    "mflops": 0.05,
    # Peak working RAM in KB needed on the MCU for layer input/output buffers.
    "arena_kb": 0.03,
    # Measured single-thread host TFLite interpreter latency in milliseconds.
    "host_latency_ms": 0.02,
    # Estimated 64 MHz Cortex-M4F inference latency in milliseconds from MACs.
    "mcu_ms_est": 0.02,
    # Total trainable weight parameters in thousands.
    "params_k": 0.02,
    # Fraction of heavy dense and convolution ops quantized to int8 activations.
    "int8_fraction": 0.05,
    # Argmax match rate between returned predictions and evaluator TFLite run.
    "tflite_agreement": 0.03,
    # Total wall-clock execution and conversion time across folds.
    "exec_time": 0.02,
}

SECTION_TITLES: tuple[str, ...] = (
    "1. Objective",
    "2. Data Contract",
    "3. Evaluation",
    "4. Environment",
    "5. Principles",
    "6. Pitfalls",
    "7. Iteration Discipline",
    "8. Output Contract Checklist",
)

STUDENT_INSTRUCTIONS_TEMPLATE = """# Task: {task_title}

## 1. Objective

Classify 1 second accelerometer windows from a Seeed XIAO nRF52840 Sense (LSM6DS3, 100 Hz, raw g units) into {num_classes} classes: {class_list}. The winning model is flashed to the board and runs under TensorFlow Lite Micro on a 64 MHz ARM Cortex-M4F with 1 MB flash shared with the application and 256 KB RAM, using portable scalar C++ reference kernels rather than ARM CMSIS-NN SIMD intrinsics (so every MAC operation directly costs CPU cycles). On the device the model runs continuously on a sliding window, so avoiding false triggers on everyday motion matters as much as recognizing gestures across shifted window timing and grip variation.

Overall score is a weighted sum of normalized metrics in three groups:

- **Prediction quality (0.68):** macro F1 ({f1_w:.2f}), balanced accuracy ({bacc_w:.2f}), worst-class recall ({wcr_w:.2f}), Null false-trigger rate ({nfpr_w:.2f}, lower is better).
- **On-device cost (0.22, scored 1.0 at zero and 0.0 at soft budget):** TFLite size KB ({kb_w:.2f}, budget {kb_budget:g}), MFLOPs ({flops_w:.2f}, budget {flops_budget:g}), activation arena KB ({arena_w:.2f}, budget {arena_budget:g}), host latency ms ({lat_w:.2f}, budget {lat_budget:g}), estimated Cortex-M4F latency ms ({mcu_w:.2f}, budget {mcu_budget:g}), parameters in thousands ({params_w:.2f}, budget {params_budget:g}).
- **Deploy hygiene (0.10):** int8 op fraction ({int8_w:.2f}), prediction-to-TFLite agreement ({agree_w:.2f}), training wall-clock time ({exec_w:.2f}).

Metrics summary:
{metrics_desc}

## 2. Data Contract

- **Inputs:**
  * `X_train` (pd.DataFrame, {train_rows} rows): {num_features} float columns `s000_x, s000_y, s000_z, ..., s{last_row:03d}_z` holding one {window_rows}x3 window in row-major time order, plus string column `recording_id`.
  * `y_train` (pd.DataFrame): single integer column `'{target_column}'` with class indices 0..{last_class} in the order {class_list}.
  * `X_val` (pd.DataFrame, {val_rows} rows): same columns as `X_train`.
- Use `recording_id` as the group key for any internal cross-validation and drop it before modelling.
- Values are raw accelerometer readings in g (`+X` toward USB-C cable edge, `+Y` toward left pin edge, `+Z` up out of the chip face). Gravity (`~1.0 g` at rest) is present and encodes board pose across stationary and hand-held rest (`Null`), palm and finger strikes (`Slap`, `Poke`), cable motions (`Flick`, `Twirl`, `Pendulum`), and direct wrist motions (`Shake`, `Punch`, `Roll`).
- **Output:** return `(y_pred_df, feedback)`:
  * `y_pred_df` (pd.DataFrame): index identical to `X_val.index`, one integer column `'{target_column}'` with the predicted class index for every row, and `y_pred_df.attrs['tflite_model']` set to the `bytes` returned by `tf.lite.TFLiteConverter(...).convert()`.
  * `feedback` (str): 4 to 8 concise lines logging the hypothesis, validation macro F1 and worst-class recall, TFLite KB, parameters, MFLOPs, int8 fraction, agreement, and runtime.

## 3. Evaluation

- **Validation protocol:** {cv_protocol} The held-out evaluation splits test continuous sliding-window conditions including shifted timing, slight wrist tilt, amplitude variation, and boundary Null handling.
- **Deploy contract (hard, checked every cycle):** the attached TFLite model must (a) accept {num_features} values as a single fixed-shape input tensor with batch 1; (b) produce {num_classes} class scores; (c) use only firmware-supported ops: {allowed_ops}; (d) be at most {max_kb:g} KB with an estimated activation arena at most {max_arena:g} KB; (e) match `y_pred_df` on at least {min_agreement:.0%} of rows when the evaluator runs your TFLite bytes directly on raw `X_val`. Any normalization or feature transformation must therefore live inside the exported graph.
- **Per-call wall-clock budget:** {timeout_s} s per call of `solution()`. {cv_budget_note}**Final evaluation budget:** {final_eval_timeout_s} s on {final_eval_rows} learning rows.
- **Resources:** CPU only, no GPU, no network.

## 4. Environment

- **Importable:** {imp_str}. Import `tensorflow` ({tf_version}) inside `solution()`, not at module top level.
- **Forbidden:** {forb_str}.
- **Shared toolkit:** `from codegen_task import toolkit as tk` is importable inside `solution()` and provides tested helpers:
{toolkit_api}
- **Response format:** begin every response with a concise 2-3 sentence plan stating the single hypothesis being tested and which score bottleneck it targets, followed by one complete Python code block defining `solution(X_train, y_train, X_val)`.

## 5. Principles

- **In-graph preprocessing with minimal op overhead:** Learn normalization statistics on the training split and either bake them into the input via `tk.norm_layer(x_tr)` or fold them analytically into the first valid-padded convolution layer's weights and biases. Keep early tensor shapes and derived operations lean, since `.tflite` flatbuffer size is the sum of quantized weights plus per-layer operator and quantization-scale metadata tables, and striding the time axis before widening or branching keeps peak activation arena small.
- **Temporal structure and amplitude summary:** Gestures differ in both peak impact magnitude and temporal ordering across the 1 second window. Choose temporal feature extractors and pooling layers that preserve both amplitude extremes and coarse temporal progression while keeping the classifier head small.
- **Full-integer quantization:** Converting weights and activations to int8 (`tk.to_int8_tflite`) shrinks model size roughly 4x and maximizes `int8_fraction`. Calibrate with representative windows covering both low-energy rest and high-g impacts.
- **Data-centric generalization at zero on-device cost:** With only a few recordings per class, training-time augmentation (`tk.augment`), realistic background negatives (`tk.hard_negatives`), and class weighting (`tk.class_weights`) improve sliding-window generalization without adding a single byte or MAC to the exported TFLite flatbuffer.

## 6. Pitfalls

- Returning predictions from the float Keras model instead of running the converted TFLite bytes (`tk.predict_tflite`) fails the agreement check.
- Training multiple models, ensembles, or teacher-student distillation pipelines inside `solution()` wastes the per-call CPU budget; train a single compact model per call.
- Running `tk.grouped_oof` inside `solution()` when `cv_folds > 1` multiplies runtime across folds; train once on `X_train` and predict `X_val` directly.
- Operators outside `ALLOWED_OPS`—notably `reduce_sum`, `reduce_std`, `reduce_variance`, and `squared_difference`—fail the deploy validator; express statistics through `MEAN`, basic arithmetic, convolutions, pooling, and dense layers.
- TensorFlow CPU `DepthwiseConv2D` requires equal row and column strides; use symmetric strides on 4D tensors or stride with standard convolution and pooling layers.
- Keras 1D convolution and pooling layers (`Conv1D`, `DepthwiseConv1D`, 1D pooling) lower to 2D TFLite kernels wrapped in extra `EXPAND_DIMS`, `RESHAPE`, and `SQUEEZE` ops that inflate flatbuffer KB, operator count, and host latency compared to expressing the network directly on 4D tensors (`batch=1`, time, singleton width, channels).
- When applying temporal sub-window pooling after strided convolutions, derive `pool_size` and `strides` from the tensor's current static time dimension (`int(h.shape[1])` or `int(h.shape[2])`) rather than hardcoding sizes that may exceed a valid-padded feature map and raise negative-dimension errors.
- `tk.augment` already includes the original training windows alongside the augmented copies, and `SparseCategoricalCrossentropy` does not support `label_smoothing` (use one-hot labels with `CategoricalCrossentropy` if smoothing).
- Leaving the Keras `Input` batch dimension unspecified causes `Reshape` and `Flatten` layers to emit dynamic shape-inference ops (`SHAPE`, `STRIDED_SLICE`, `PACK`) that inflate flatbuffer KB; declare a fixed batch size of 1 on `Input`, avoid dynamic `None` dimensions, and return integer predictions aligned to `X_val.index`.

## 7. Iteration Discipline

- **Cycle 0:** start from a simple working baseline that satisfies the deploy contract end to end.
- **One lever per cycle:** change one aspect at a time—in-graph normalization, quantization, architecture family, temporal pooling, data augmentation, calibration, or footprint pruning—and keep changes that improve validation score or cut footprint at equal accuracy.
- **Read the score breakdown:** target whichever component (F1, worst-class recall, Null false-trigger rate, or KB/MFLOPs/arena) is currently dragging down `overall_score`.
- **Use `feedback` as a concise lab log:** record your hypothesis, validation metrics, model size, MFLOPs, int8 fraction, and runtime every cycle.

## 8. Output Contract Checklist

- [ ] Response starts with a concise 2-3 sentence hypothesis plan before the Python block.
- [ ] Signature is `solution(X_train, y_train, X_val)` returning `(y_pred_df, feedback)`.
- [ ] `y_pred_df` index equals `X_val.index`; one integer column `'{target_column}'`; no NaN.
- [ ] `y_pred_df.attrs['tflite_model']` holds the converted TFLite `bytes`.
- [ ] Graph accepts {num_features} values with static batch 1, outputs {num_classes} scores, uses only allowed ops, and fits within {max_kb:g} KB model size and {max_arena:g} KB arena.
- [ ] Predictions come from running those exact TFLite bytes on raw `X_val`.
- [ ] Single-model training and conversion finish well inside {timeout_s} s.
"""

STUDENT_INITIAL_PROMPT = (
    "Start with a concise 1-2 sentence plan, then provide the Cycle 0"
    " `solution(X_train, y_train, X_val)` implementing the v1 baseline"
    " (`Flatten -> Dense(32, relu) -> Dense(16, relu)"
    " -> Dense(k, softmax)` trained on raw `X_train` without in-graph"
    " normalization, quantization, or augmentation, converted with"
    " `tk.to_float_tflite`) so Cycle 0 measures the unmodified v1 starting"
    " point before refinement."
)

STUDENT_REFINE_INSTRUCTION_OPTIONS: list[str] = [
    (
        "In-Graph Standardization and Full Int8 Quantization: Bake per-axis"
        " training statistics into the model input with tk.norm_layer(x_tr)"
        " or fold them analytically into the first valid-padded convolution"
        " layer's weights and biases, and convert the network to full int8"
        " with tk.to_int8_tflite to shrink flash size and earn the"
        " int8_fraction bonus."
    ),
    (
        "Temporal Convolutions Over Native 4D Tensors: Replace wide"
        " flattened Dense layers with strided temporal 2D convolutions over"
        " reshaped 4D time-by-axis tensors (avoiding 1D Keras wrapper ops)"
        " so shift invariance improves and parameter count drops sharply."
    ),
    (
        "Factorized Depthwise-Separable Convolutions: Separate spatial-temporal"
        " filtering from pointwise channel mixing with folded batch"
        " normalization (keeping symmetric 2D strides where required by"
        " TensorFlow CPU) to cut KB, MFLOPs, and activation arena."
    ),
    (
        "Chronology-Preserving Multi-Statistic Pooling: Combine complementary"
        " average and max pooling across the downsampled time axis—globally"
        " or over coarse temporal sub-windows derived dynamically from the"
        " feature map's static time dimension—so sustained posture, sharp"
        " impact spikes, and early-vs-late sequence ordering reach the"
        " classifier head."
    ),
    (
        "Physics-Aware Data Augmentation: Expand the small training set with"
        " tk.augment and physically grounded within-class mixing that"
        " preserves resting gravity pose and aligns dynamic impulses rather"
        " than smearing single impacts into artificial double peaks."
    ),
    (
        "Null Boundary Hardening and Prior Preservation: Synthesize realistic"
        " low-energy background and slow-tilt windows with tk.hard_negatives"
        " while ensuring synthetic negative counts do not dilute the Null"
        " class weight in the training loss."
    ),
    (
        "In-Graph Kinematic Channels vs Lean Stem: Test whether lightweight"
        " in-graph motion-magnitude or gravity-centered channels improve"
        " fine-grained class separation enough to justify their extra"
        " flatbuffer operators and early activation arena over a direct"
        " strided stem."
    ),
    (
        "Shared-Stem Multi-Scale Temporal Receptive Fields: Capture both brief"
        " high-jerk strikes and slower oscillatory or rotational motions by"
        " downsampling first through a shared strided stem before applying"
        " multi-scale temporal kernels at the compressed bottleneck."
    ),
    (
        "Lightweight Channel Recalibration: Experiment with compact"
        " channel-wise gating using global average pooling, a narrow"
        " projection, logistic activation, and elementwise multiplication to"
        " weight pose vs impulse channels at minimal compute cost."
    ),
    (
        "Static Shapes, Operator Count, and Footprint Pruning: Inspect"
        " tk.inspect(tflite) to eliminate dynamic shape-inference ops via"
        " fixed batch_size=1 inputs, fold normalization into weights,"
        " consolidate redundant operator nodes (since each conv/pooling/concat"
        " layer adds per-channel flatbuffer metadata tables), and trim channel"
        " widths where validation F1 saturates."
    ),
    (
        "Regularization and Learning-Rate Scheduling: Improve small-data"
        " generalization and worst-class recall using dropout, weight decay,"
        " cosine learning-rate scheduling, or one-hot cross-entropy with"
        " mild label smoothing."
    ),
    (
        "Quantization Calibration Coverage: Construct a representative"
        " calibration set that spans quiet resting poses, subtle rotations,"
        " and high-amplitude gesture windows without extreme synthetic"
        " outliers distorting int8 step scales."
    ),
    (
        "Decision Margin and Output Logit Calibration: Refine class boundaries"
        " after training—especially between low-energy motion and Null or"
        " confused gesture pairs—through subtle output bias or temperature"
        " calibration baked directly into the final layer weights at zero"
        " extra inference cost."
    ),
    (
        "Small-Sample Convergence and Class Balance: Tune training epoch"
        " budget, batch size, augmentation multiplicity, and per-class sample"
        " weighting to lift worst_class_recall without overfitting or"
        " increasing execution time."
    ),
]

_FORBIDDEN_RECIPE_PATTERNS = (
    re.compile(r"\b(lr|learning_rate)\s*=\s*[0-9.e-]+", re.IGNORECASE),
    re.compile(r"\b(epochs|filters|units|kernel_size)\s*=\s*[0-9]+",
               re.IGNORECASE),
    re.compile(r"\b(dropout|l2)\s*=\s*[0-9.]+", re.IGNORECASE),
    re.compile(r"\be\.g\.", re.IGNORECASE),
    re.compile(r"\bfor example\b", re.IGNORECASE),
)


def assert_no_recipes(text: str) -> None:
  """Verifies prompts give principles rather than recipes or 'e.g.' hints."""
  for pattern in _FORBIDDEN_RECIPE_PATTERNS:
    match = pattern.search(text)
    if match:
      raise AssertionError(
          f"Forbidden recipe or hint pattern: {match.group(0)!r}"
      )


def build_student_instructions(
    *,
    class_names: Sequence[str],
    train_rows: int,
    val_rows: int,
    final_eval_rows: int,
    timeout_s: int = RUN_DEFAULTS["timeout_s"],
    final_eval_timeout_s: int = RUN_DEFAULTS["final_eval_timeout_s"],
    metric_weights: Mapping[str, float] | None = None,
    window_rows: int = data_loader.WINDOW_ROWS,
    tf_version: str = "2.15",
    cv_folds: int = 0,
    prior_insights: str = "",
) -> str:
  """Formats the student instructions for the current dataset.

  `prior_insights` is an optional, already recipe-scrubbed markdown block
  (see knowledge.prompt_section) appended after the checklist.
  """
  w = dict(metric_weights or DEFAULT_METRIC_WEIGHTS)
  if cv_folds and cv_folds > 1:
    cv_protocol = (
        f"The evaluator calls `solution()` {cv_folds} times per cycle, once"
        " per stratified fold over all learning recordings, and scores the"
        " concatenated out-of-fold predictions. Each call sees a different"
        " `X_train`/`X_val` pair, so do not run your own K-fold inside"
        " `solution()`; a cheap internal hold-out for early stopping is fine."
    )
    cv_budget_note = (
        f"Because of the {cv_folds} folds, keep a single call well under the"
        " budget. ")
  else:
    cv_protocol = (
        "Hold-out `X_val` during cycles. It is small, so run grouped K-fold on"
        " `X_train` internally, report it in `feedback`, and trust it over"
        " the hold-out when they disagree.")
    cv_budget_note = ""
  text = STUDENT_INSTRUCTIONS_TEMPLATE.format(
      task_title=TASK_TITLE,
      num_classes=len(class_names),
      last_class=len(class_names) - 1,
      class_list=", ".join(f"`{c}`" for c in class_names),
      f1_w=w.get("f1_macro", 0.0),
      bacc_w=w.get("balanced_accuracy", 0.0),
      wcr_w=w.get("worst_class_recall", 0.0),
      nfpr_w=w.get("null_fpr", 0.0),
      kb_w=w.get("model_kb", 0.0),
      flops_w=w.get("mflops", 0.0),
      arena_w=w.get("arena_kb", 0.0),
      lat_w=w.get("host_latency_ms", 0.0),
      mcu_w=w.get("mcu_ms_est", 0.0),
      params_w=w.get("params_k", 0.0),
      int8_w=w.get("int8_fraction", 0.0),
      agree_w=w.get("tflite_agreement", 0.0),
      exec_w=w.get("exec_time", 0.0),
      kb_budget=deploy.SOFT_MODEL_KB,
      flops_budget=deploy.SOFT_MFLOPS,
      arena_budget=deploy.SOFT_ARENA_KB,
      lat_budget=deploy.SOFT_HOST_LATENCY_MS,
      mcu_budget=deploy.SOFT_MCU_MS,
      params_budget=deploy.SOFT_PARAMS_K,
      metrics_desc=scoring.get_metrics_description(w).strip(),
      train_rows=train_rows,
      val_rows=val_rows,
      final_eval_rows=final_eval_rows,
      num_features=window_rows * data_loader.NUM_AXES,
      window_rows=window_rows,
      last_row=window_rows - 1,
      target_column=data_loader.TARGET_COLUMN,
      allowed_ops=", ".join(f"`{o}`" for o in sorted(deploy.ALLOWED_OPS)),
      max_kb=deploy.MAX_MODEL_KB,
      max_arena=deploy.MAX_ARENA_KB,
      min_agreement=deploy.MIN_AGREEMENT,
      timeout_s=timeout_s,
      final_eval_timeout_s=final_eval_timeout_s,
      cv_protocol=cv_protocol,
      cv_budget_note=cv_budget_note,
      imp_str="`numpy`, `pandas`, `scipy`, `sklearn`, `tensorflow`,"
              " `codegen_task.toolkit`",
      toolkit_api="\n".join("  " + l for l in
                             toolkit.api_summary().splitlines()),
      tf_version=tf_version,
      forb_str="`torch`, `jax`, `lightgbm`, `xgboost`, multiprocessing pools",
  )
  if prior_insights:
    text = text.rstrip() + "\n\n" + prior_insights.strip() + "\n"
  return text


# ============================================================================
# Reference baseline: Vanilla Original Playground float32 MLP
# (adapted from Python/training_widget.py to the solution() contract)
# ============================================================================

BASELINE_SOLUTION = '''```python
import time
import numpy as np
import pandas as pd


def solution(X_train, y_train, X_val):
  """Vanilla Original Playground baseline: float32 Flatten + Dense(32, 16) MLP."""
  import tensorflow as tf
  from codegen_task import toolkit as tk

  t0 = time.time()
  tk.set_determinism(0)
  d = tk.split_xy(X_train, y_train, X_val)
  x_tr, y_tr, x_va, rows, k = (
      d["x_tr"], d["y_tr"], d["x_va"], d["rows"], d["n_classes"]
  )

  # Original Playground architecture from Python/training_widget.py:
  # raw unnormalized accelerometer input, float32 Dense(32) -> Dense(16) MLP,
  # trained without augmentation or hard negatives.
  inp = tf.keras.Input(shape=(rows, 3))
  h = tf.keras.layers.Flatten()(inp)
  h = tf.keras.layers.Dense(32, activation="relu")(h)
  h = tf.keras.layers.Dense(16, activation="relu")(h)
  out = tf.keras.layers.Dense(k, activation="softmax")(h)
  model = tf.keras.Model(inp, out)
  model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
  model.fit(x_tr, y_tr, epochs=35, batch_size=16, verbose=0)

  tflite = tk.to_float_tflite(model)
  preds = tk.predict_tflite(tflite, x_va).argmax(1)
  info = tk.inspect(tflite)
  problems = tk.check_contract(tflite, k)
  feedback = (
      "Vanilla Original Playground baseline: float32 Flatten->Dense(32)->Dense(16)"
      " on raw unnormalized input (no int8, no augmentation).\\n"
      f"tflite={info['model_kb']:.1f} KB mflops={info['mflops']:.3f}"
      f" arena={info['arena_kb']:.1f} KB int8={info['int8_fraction']:.2f}"
      f" contract={'ok' if not problems else problems}"
      f" runtime={time.time()-t0:.1f}s"
  )
  return tk.make_result(X_val, y_train, preds, tflite, feedback)
```'''


# ============================================================================
# Cross-run memory: archetypes, insights, frontier, learned levers
# ============================================================================

INSIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "solutions", "insights.json")

MAX_INSIGHTS = 10
MAX_LEARNED_LEVERS = 6

# Masks hyperparameter recipes and 'e.g.' hints so injected text stays clean.
_RECIPE_RE = re.compile(
    r"\b(lr|learning_rate|epochs|filters|units|kernel_size|dropout|l2|"
    r"batch_size|n_estimators|max_depth|alpha)\s*=\s*[0-9.e-]+",
    re.IGNORECASE)
_HINT_PAREN_RE = re.compile(
    r"\s*\((?:e\.g\.|for example|such as)\b[^)]*\)",
    re.IGNORECASE,
)
_EXAMPLE_PHRASE_RE = re.compile(
    r"\b(?:e\.g\.,?\s*|for example,?\s*)",
    re.IGNORECASE,
)

# Starting directions handed to unseeded students so cycle 0 covers
# different simple model families and leaves headroom to climb.
ARCHETYPES: tuple[str, ...] = (
    "Start with a compact in-graph normalized temporal convolution baseline"
    " with global pooling and int8 conversion.",
    "Start with a compact temporal projection and pooled MLP baseline"
    " with in-graph normalization and int8 conversion.",
    "Start with a lightweight strided temporal convolution stem followed by"
    " complementary global pooling and int8 conversion.",
    "Start with a compact depthwise-separable temporal convolution baseline"
    " on standardized accelerometer axes with balanced class weights.",
)


def _load() -> dict[str, Any]:
  if os.path.isfile(INSIGHTS_PATH):
    with open(INSIGHTS_PATH, encoding="utf-8") as f:
      return json.load(f)
  return {"runs": [], "learned_levers": []}


def _save(data: Mapping[str, Any]) -> None:
  os.makedirs(os.path.dirname(INSIGHTS_PATH), exist_ok=True)
  with open(INSIGHTS_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)


def scrub(text: str) -> str:
  """Removes hyperparameter assignments and 'e.g.' hints from prompt text."""
  cleaned = _RECIPE_RE.sub(lambda m: f"{m.group(1)}=…", text or "")
  cleaned = _HINT_PAREN_RE.sub("", cleaned)
  cleaned = _EXAMPLE_PHRASE_RE.sub("", cleaned)
  return re.sub(r"  +", " ", cleaned).strip()


def record_run(
    run_id: str,
    summary: Mapping[str, Any],
    analysis: Mapping[str, Any] | None,
    students: Mapping[str, Any],
    dataset_fingerprint: str,
) -> None:
  """Appends one run's distilled lessons to the insights store."""
  data = _load()
  art = summary.get("artifact", {})
  tm = summary.get("test_metrics", {})
  attempts = []
  for name, st in students.items():
    for r in getattr(st, "cycle_results", []):
      if r.overall_score is None:
        continue
      m = r.metrics or {}
      attempts.append({
          "student": name, "cycle": r.cycle,
          "score": float(r.overall_score),
          "f1": float(m.get("f1_macro", float("nan"))),
          "kb": float(m.get("model_kb", float("nan"))),
          "mflops": float(m.get("mflops", float("nan"))),
          "plan": (r.plan or "")[:300],
      })

  def _fmt_field(val: Any) -> str:
    if isinstance(val, (list, tuple)):
      return " ".join(scrub(str(x)).strip() for x in val if str(x).strip())
    return scrub(str(val or ""))

  raw_next = (analysis or {}).get("next_experiments", "")
  next_items = (
      [scrub(str(x)).strip() for x in raw_next if str(x).strip()]
      if isinstance(raw_next, (list, tuple))
      else re.split(r"(?<=[.;])\s+|\n+", _fmt_field(raw_next))
  )
  entry = {
      "run_id": run_id,
      "dataset": dataset_fingerprint,
      "test_overall": float(summary.get("test_overall_score", float("nan"))),
      "test_f1": float(tm.get("f1_macro", float("nan"))),
      "model_kb": float(art.get("model_kb", float("nan"))),
      "mflops": float(art.get("mflops", float("nan"))),
      "arena_kb": float(art.get("arena_kb", float("nan"))),
      "retrain_delta": float(summary.get("retrain_delta", float("nan"))),
      "ops": list(art.get("ops", [])),
      "model_description": _fmt_field((analysis or {}).get("model_description")),
      "key_aspects": _fmt_field((analysis or {}).get("key_aspects")),
      "improvements": _fmt_field((analysis or {}).get("improvements_from_baseline")),
      "risks": _fmt_field((analysis or {}).get("regressions_or_risks")),
      "next_experiments": _fmt_field(raw_next),
      "attempts": attempts[-40:],
  }
  data["runs"].append(entry)
  if next_items:
    levers = [scrub(str(x)) for x in data.get("learned_levers", [])]
    for piece in next_items:
      piece = scrub(piece).strip(" -•\t[]'\"")
      if len(piece) > 30 and piece not in levers:
        levers.append(piece)
    data["learned_levers"] = levers[-MAX_LEARNED_LEVERS:]
  _save(data)


def learned_levers() -> list[str]:
  """Refinement directions proposed by the analyzer in earlier runs."""
  out = []
  for raw in _load().get("learned_levers", []):
    cleaned = scrub(str(raw))
    if cleaned:
      out.append(f"Learned direction: {cleaned}")
  return out


def frontier(runs: Sequence[Mapping[str, Any]] | None = None) -> list[str]:
  """Best score seen at a few size tiers across all recorded attempts."""
  data = _load() if runs is None else {"runs": runs}
  pts = []
  for r in data.get("runs", []):
    for a in r.get("attempts", []):
      if a.get("kb") == a.get("kb") and a.get("f1") == a.get("f1"):
        pts.append((a["kb"], a["f1"], a["mflops"]))
  if not pts:
    return []
  out = []
  for cap in (4, 8, 16, 32):
    tier = [p for p in pts if p[0] <= cap]
    if tier:
      best = max(tier, key=lambda p: p[1])
      out.append(f"<= {cap} KB: best val F1 {best[1]:.2f} at {best[0]:.1f} KB,"
                 f" {best[2]:.3f} MFLOPs")
  return out


def prompt_section(dataset_fingerprint: str) -> str:
  """Bounded numeric frontier summary from earlier runs without recipe leaks."""
  data = _load()
  runs = data.get("runs", [])
  if not runs:
    return ""
  lines = ["## 9. What Previous Runs Learned", ""]
  same = [r for r in runs if r.get("dataset") == dataset_fingerprint]
  lines.append(
      f"{len(runs)} earlier runs recorded ({len(same)} on this exact dataset)."
      " Use the quantitative frontier below to gauge achievable accuracy-cost"
      " trade-offs while discovering architectures from first principles.")
  lines.append("")
  fr = frontier(runs)
  if fr:
    lines.append("Frontier so far (validation F1 by size tier):")
    lines += [f"- {f}" for f in fr]
    lines.append("")
  for r in runs[-MAX_INSIGHTS:]:
    lines.append(
        f"- Run `{r['run_id']}`: test F1 {r['test_f1']:.2f}, overall"
        f" {r['test_overall']:.3f}, {r['model_kb']:.1f} KB,"
        f" {r['mflops']:.3f} MFLOPs, retrain delta {r['retrain_delta']:+.3f}."
    )
    risk = scrub(str(r.get("risks", "")))
    if risk:
      lines.append(f"  - Watch-outs: {risk}")
  lines.append("")
  return "\n".join(lines)


def cycle_feedback(
    students: Mapping[str, Any],
    refine_options: Sequence[str],
) -> str:
  """Extra per-cycle text: Pareto frontier, bottleneck diagnosis, untried levers."""
  pts, notes, student_diags = [], [], []
  w = DEFAULT_METRIC_WEIGHTS
  for st_name, st in students.items():
    best_r = None
    last_r = None
    for r in getattr(st, "cycle_results", []):
      if r.overall_score is None or not (r.overall_score > -1e9):
        continue
      last_r = r
      m = r.metrics or {}
      if "model_kb" in m and "f1_macro" in m:
        pts.append((
            float(m["model_kb"]),
            float(m["f1_macro"]),
            float(r.overall_score),
            float(m.get("mflops", 0.0)),
            float(m.get("worst_class_recall", 0.0)),
        ))
      notes.append(
          f"{r.plan or ''} {getattr(r, 'execution_feedback', '') or ''}".lower()
      )
      if best_r is None or float(r.overall_score) > float(best_r.overall_score):
        best_r = r
    if best_r is not None:
      bm = best_r.metrics or {}
      f1 = float(bm.get("f1_macro", 0.0))
      wcr = float(bm.get("worst_class_recall", 0.0))
      nfpr = float(bm.get("null_fpr", 0.0))
      kb = float(bm.get("model_kb", 0.0))
      mf = float(bm.get("mflops", 0.0))
      ar = float(bm.get("arena_kb", 0.0))
      lat = float(bm.get("host_latency_ms", 0.0))
      f1_loss = w["f1_macro"] * max(0.0, 1.0 - f1)
      wcr_loss = w["worst_class_recall"] * max(0.0, 1.0 - wcr)
      nfpr_loss = w["null_fpr"] * min(1.0, max(0.0, nfpr))
      kb_loss = w["model_kb"] * min(1.0, max(0.0, kb) / deploy.SOFT_MODEL_KB)
      mf_loss = w["mflops"] * min(1.0, max(0.0, mf) / deploy.SOFT_MFLOPS)
      ar_loss = w["arena_kb"] * min(1.0, max(0.0, ar) / deploy.SOFT_ARENA_KB)
      lat_loss = w["host_latency_ms"] * min(
          1.0, max(0.0, lat) / deploy.SOFT_HOST_LATENCY_MS
      )
      gaps = sorted(
          [
              (f1_loss, f"f1={f1:.3f} (-{f1_loss:.3f} pts)"),
              (wcr_loss, f"wcr={wcr:.3f} (-{wcr_loss:.3f} pts)"),
              (kb_loss, f"kb={kb:.1f}KB (-{kb_loss:.3f} pts)"),
              (mf_loss, f"mflops={mf:.3f} (-{mf_loss:.3f} pts)"),
              (ar_loss, f"arena={ar:.1f}KB (-{ar_loss:.3f} pts)"),
              (lat_loss, f"lat={lat:.2f}ms (-{lat_loss:.3f} pts)"),
              (nfpr_loss, f"null_fpr={nfpr:.3f} (-{nfpr_loss:.3f} pts)"),
          ],
          key=lambda x: x[0],
          reverse=True,
      )
      top_gaps = ", ".join(g[1] for g in gaps[:2] if g[0] > 0.003)
      reg_note = ""
      if last_r is not None and last_r.cycle > best_r.cycle:
        reg_note = (
            f" | Last C{last_r.cycle}={float(last_r.overall_score):.4f}"
            f" below C{best_r.cycle} best"
        )
      student_diags.append(
          f"- `{st_name}` best (C{best_r.cycle}): score={best_r.overall_score:.4f}"
          f" (F1={f1:.3f}, wcr={wcr:.3f}, KB={kb:.1f}, MFLOPs={mf:.3f},"
          f" arena={ar:.1f}KB) | Biggest score drag:"
          f" {top_gaps or 'near ceiling'}{reg_note}"
      )
  lines = []
  if pts:
    best = max(pts, key=lambda p: p[2])
    small = min(pts, key=lambda p: p[0])
    lines.append(
        f"Run frontier: best overall {best[2]:.4f} (F1 {best[1]:.3f}, wcr"
        f" {best[4]:.3f} at {best[0]:.1f} KB, {best[3]:.3f} MFLOPs); smallest"
        f" model {small[0]:.1f} KB (F1 {small[1]:.3f}, overall {small[2]:.4f})."
    )
    tiers = []
    for cap in (6, 8, 12, 16):
      tier = [p for p in pts if p[0] <= cap]
      if tier:
        tb = max(tier, key=lambda p: p[2])
        tiers.append(
            f"<={cap}KB: {tb[2]:.4f} (F1 {tb[1]:.3f} @ {tb[0]:.1f}KB/{tb[3]:.3f}MFLOPs)"
        )
    if tiers:
      lines.append("Pareto tiers in this run: " + " | ".join(tiers))
  if student_diags:
    lines.append("Per-student best and score bottleneck diagnosis:")
    lines.extend(student_diags)
  keywords = {
      ("segment", "sub-window", "subwindow", "chunk"): (
          "temporal sub-window pooling"
      ),
      ("blend", "mixup", "mix_x", "within-class"): (
          "aligned within-class mixing"
      ),
      ("pose", "gravity", "impulse"): "gravity-pose vs impulse augmentation",
      ("gating", "recalibrat", "squeeze"): "channel recalibration",
      ("cosine", "schedule"): "cosine learning-rate schedule",
      ("smooth",): "label smoothing",
      ("negative", "hard_neg"): "background Null boundary hardening",
      ("depthwise", "separable"): "depthwise-separable convolutions",
      ("calibrat", "temperature", "margin"): (
          "output logit or quantization calibration"
      ),
  }
  text = " ".join(notes)
  untried = [
      v for ks, v in keywords.items() if not any(k in text for k in ks)
  ]
  if untried:
    lines.append("Levers no one in this run has tried yet: "
                 + ", ".join(untried) + ".")
  del refine_options
  return "\n".join(lines)
