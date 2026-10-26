"""Deploy contract for micro_gesture: artifact analysis, validator, metrics.

The classroom scores `solution()` on prediction quality only. This module adds
the on-device objectives without touching the library:

* The student attaches the serialized TFLite flatbuffer to
  `y_pred_df.attrs["tflite_model"]`.
* `deploy_validator` runs in the parent thread right before scoring. It pops
  the bytes, analyses the graph (size, MACs, arena estimate, host latency,
  op set, I/O shapes), checks hard budgets, and verifies that running the
  TFLite model on `X_val` reproduces the returned predictions.
* Measurements are stashed in a thread-local; the registered cost metrics read
  them when `scoring.score_predictions` runs on the same thread.
"""

from __future__ import annotations

import dataclasses
import logging
import argparse
import glob
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from codegen_classroom import metrics as metric_lib
from codegen_classroom import scoring

from codegen_task import data_loader

logger = logging.getLogger(__name__)

ATTR_KEY = "tflite_model"

# Hard limits: a solution violating any of these is rejected outright.
MAX_MODEL_KB = 64.0
MAX_ARENA_KB = 40.0
MIN_AGREEMENT = 0.95

# Soft budgets: cost metrics score 1.0 at zero and 0.0 at the budget.
SOFT_MODEL_KB = 16.0
SOFT_MFLOPS = 0.20
SOFT_ARENA_KB = 6.0
SOFT_HOST_LATENCY_MS = 0.05
SOFT_PARAMS_K = 6.0
SOFT_MCU_MS = 18.0

# Cortex-M4F at 64 MHz running TFLM reference int8 kernels. Calibrated on
# 2026-10-22: 119k MACs measured 25.9 ms on the XIAO (LAT: lines).
MCU_HZ = 64e6
MCU_MACS_PER_CYCLE = 0.075
# TFLM allocates per-tensor headers and kernel scratch beyond live activation
# bytes; measured ARENA: was about 2.2x the activation estimate.
ARENA_OVERHEAD = 2.2

# Ops the firmware's MicroMutableOpResolver registers. Keep in sync with
# Arduino/Codegen/Codegen.ino.
ALLOWED_OPS: frozenset[str] = frozenset({
    "FULLY_CONNECTED", "CONV_2D", "DEPTHWISE_CONV_2D", "MAX_POOL_2D",
    "AVERAGE_POOL_2D", "RESHAPE", "SOFTMAX", "QUANTIZE", "DEQUANTIZE",
    "MEAN", "ADD", "SUB", "MUL", "RELU", "RELU6", "LOGISTIC", "TANH",
    "PAD", "EXPAND_DIMS", "SQUEEZE", "CONCATENATION", "STRIDED_SLICE",
    "L2_NORMALIZATION", "LEAKY_RELU", "HARD_SWISH", "ABS", "SQRT", "SQUARE",
    "MAXIMUM", "MINIMUM", "REDUCE_MAX", "ARG_MAX", "TRANSPOSE", "SHAPE",
    "SPLIT", "GATHER", "SLICE", "CAST", "PACK", "UNPACK",
})

# Metrics read from the validator's artifact analysis (not from y_pred).
ARTIFACT_METRICS: tuple[str, ...] = (
    "model_kb", "mflops", "arena_kb", "host_latency_ms", "params_k",
    "mcu_ms_est", "int8_fraction", "tflite_agreement",
)
# Metrics computed from (y_true, y_pred) beyond the library's built-ins.
LABEL_METRICS: tuple[str, ...] = ("worst_class_recall", "null_fpr")
_null_index: int | None = None

_tls = threading.local()

# Every artifact that passed validation, keyed by a hash of the predictions
# it produced, so the runner can export exactly the model the final
# evaluation scored instead of re-fitting (training is not bit-reproducible).
_STORE_LOCK = threading.Lock()
_ARTIFACTS: dict[str, tuple[bytes, "ArtifactStats"]] = {}
_ARTIFACT_LIMIT = 400


def prediction_key(index, labels) -> str:
  """Stable key for a set of predictions (index and integer labels)."""
  import hashlib  # pylint: disable=import-outside-toplevel
  h = hashlib.sha1()
  h.update(",".join(str(i) for i in index).encode())
  h.update(b"|")
  h.update(",".join(str(int(v)) for v in labels).encode())
  return h.hexdigest()


def stored_artifact(key: str) -> tuple[bytes, "ArtifactStats"] | None:
  with _STORE_LOCK:
    return _ARTIFACTS.get(key)


@dataclasses.dataclass
class ArtifactStats:
  """Measurements of one TFLite artifact."""

  model_kb: float
  mflops: float
  arena_kb: float
  host_latency_ms: float
  ops: frozenset[str]
  input_shape: tuple[int, ...]
  input_dtype: str
  output_shape: tuple[int, ...]
  output_dtype: str
  quantized: bool
  params_k: float
  mcu_ms_est: float
  int8_fraction: float
  agreement: float = float("nan")

  @property
  def tflite_agreement(self) -> float:
    return self.agreement

  def as_dict(self) -> dict[str, Any]:
    d = dataclasses.asdict(self)
    d["ops"] = sorted(self.ops)
    return d


def _tensor_bytes(detail: Mapping[str, Any]) -> int:
  return int(np.prod(detail["shape"])) * np.dtype(detail["dtype"]).itemsize


def analyze_tflite(model_bytes: bytes, latency_runs: int = 20) -> ArtifactStats:
  """Computes size, MACs, arena estimate, latency, and I/O facts."""
  import tensorflow as tf  # pylint: disable=import-outside-toplevel

  # Disable XNNPACK: with the delegate applied, float graphs collapse into a
  # single "DELEGATE" node and the real op set is hidden.
  interp = tf.lite.Interpreter(
      model_content=model_bytes, num_threads=1,
      experimental_op_resolver_type=(
          tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
      ),
  )
  interp.allocate_tensors()
  tensors = {d["index"]: d for d in interp.get_tensor_details()}
  ops = interp._get_ops_details()  # pylint: disable=protected-access
  inp = interp.get_input_details()[0]
  out = interp.get_output_details()[0]

  graph_inputs = {d["index"] for d in interp.get_input_details()}
  activations = set(graph_inputs)
  for op in ops:
    activations.update(int(i) for i in op["outputs"])

  macs = 0
  arena = 0
  for op in ops:
    name = op["op_name"]
    outs = [tensors[int(i)] for i in op["outputs"] if int(i) in tensors]
    ins = [tensors[int(i)] for i in op["inputs"] if int(i) in tensors]
    out_elems = sum(int(np.prod(d["shape"])) for d in outs)
    if name == "FULLY_CONNECTED" and len(ins) >= 2:
      macs += out_elems * int(ins[1]["shape"][-1])
    elif name == "CONV_2D" and len(ins) >= 2:
      w = ins[1]["shape"]  # (cout, kh, kw, cin)
      macs += out_elems * int(np.prod(w[1:]))
    elif name == "DEPTHWISE_CONV_2D" and len(ins) >= 2:
      w = ins[1]["shape"]  # (1, kh, kw, cout)
      macs += out_elems * int(w[1]) * int(w[2])
    elif name == "BATCH_MATMUL" and len(ins) >= 2:
      macs += out_elems * int(ins[0]["shape"][-1])
    else:
      macs += out_elems
    live = [d for d in ins + outs if d["index"] in activations]
    arena = max(arena, sum(_tensor_bytes(d) for d in live))

  x = np.zeros(inp["shape"], dtype=inp["dtype"])
  interp.set_tensor(inp["index"], x)
  interp.invoke()
  t0 = time.perf_counter()
  for _ in range(latency_runs):
    interp.invoke()
  latency_ms = (time.perf_counter() - t0) / latency_runs * 1e3

  op_names = frozenset(op["op_name"] for op in ops)
  heavy = [op for op in ops if op["op_name"] in
           ("FULLY_CONNECTED", "CONV_2D", "DEPTHWISE_CONV_2D")]
  int8_heavy = [
      op for op in heavy
      if np.dtype(tensors[int(op["inputs"][0])]["dtype"]) == np.int8
  ]
  int8_fraction = len(int8_heavy) / len(heavy) if heavy else 0.0
  quantized = int8_fraction == 1.0 and bool(heavy)
  weight_ids = {int(i) for op in heavy for i in op["inputs"][1:]
                if int(i) in tensors}
  params = sum(int(np.prod(tensors[i]["shape"])) for i in weight_ids)
  return ArtifactStats(
      model_kb=len(model_bytes) / 1024.0,
      mflops=2.0 * macs / 1e6,
      arena_kb=arena * ARENA_OVERHEAD / 1024.0,
      host_latency_ms=latency_ms,
      ops=op_names,
      input_shape=tuple(int(v) for v in inp["shape"]),
      input_dtype=np.dtype(inp["dtype"]).name,
      output_shape=tuple(int(v) for v in out["shape"]),
      output_dtype=np.dtype(out["dtype"]).name,
      quantized=quantized,
      params_k=params / 1000.0,
      mcu_ms_est=macs / (MCU_HZ * MCU_MACS_PER_CYCLE) * 1e3,
      int8_fraction=int8_fraction,
  )


def run_tflite(model_bytes: bytes, features: np.ndarray) -> np.ndarray:
  """Runs the artifact row by row on raw features and returns class scores."""
  import tensorflow as tf  # pylint: disable=import-outside-toplevel

  interp = tf.lite.Interpreter(
      model_content=model_bytes, num_threads=1,
      experimental_op_resolver_type=(
          tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
      ),
  )
  interp.allocate_tensors()
  inp = interp.get_input_details()[0]
  out = interp.get_output_details()[0]
  in_scale, in_zp = inp["quantization"]
  out_scale, out_zp = out["quantization"]
  scores = []
  for row in features:
    x = row.astype(np.float32).reshape(inp["shape"])
    if np.dtype(inp["dtype"]) == np.int8:
      x = np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
    elif np.dtype(inp["dtype"]) == np.uint8:
      x = np.clip(np.round(x / in_scale + in_zp), 0, 255).astype(np.uint8)
    interp.set_tensor(inp["index"], x)
    interp.invoke()
    y = interp.get_tensor(out["index"]).astype(np.float32).reshape(-1)
    if np.dtype(out["dtype"]) in (np.int8, np.uint8):
      y = (y - out_zp) * out_scale
    scores.append(y)
  return np.stack(scores)


def current_stats() -> ArtifactStats | None:
  """Stats recorded by the most recent validator call on this thread."""
  return getattr(_tls, "stats", None)


def current_model_bytes() -> bytes | None:
  return getattr(_tls, "model_bytes", None)


def _artifact_metric(name: str):
  def _read(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    del y_true, y_pred
    stats = current_stats()
    if stats is None:
      raise scoring.ScoreError(
          f"Metric '{name}' requires deploy_validator to run first."
      )
    return float(getattr(stats, name))
  return _read


def worst_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Recall of the worst-served class; exposes a class the model ignores."""
  yt = np.asarray(y_true).reshape(-1).astype(int)
  yp = np.asarray(y_pred).reshape(-1).astype(int)
  recalls = [float(np.mean(yp[yt == c] == c)) for c in np.unique(yt)]
  return min(recalls) if recalls else 0.0


def null_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """Fraction of true Null windows predicted as a gesture (false triggers)."""
  if _null_index is None:
    return 0.0
  yt = np.asarray(y_true).reshape(-1).astype(int)
  yp = np.asarray(y_pred).reshape(-1).astype(int)
  mask = yt == _null_index
  if not mask.any():
    return 0.0
  return float(np.mean(yp[mask] != _null_index))


def _budget_normalizer(budget: float):
  return lambda v: float(max(0.0, 1.0 - max(0.0, v) / budget))


def configure(class_names: Sequence[str]) -> None:
  """Tells label metrics which class index is the Null / background class."""
  global _null_index  # pylint: disable=global-statement
  _null_index = class_names.index("Null") if "Null" in class_names else None


def register_metrics() -> None:
  """Registers the on-device and robustness metrics with the scorer."""
  lower = {
      "model_kb": ("TFLite flatbuffer size in KB", SOFT_MODEL_KB),
      "mflops": ("Estimated MFLOPs per inference", SOFT_MFLOPS),
      "arena_kb": ("Estimated peak activation memory in KB", SOFT_ARENA_KB),
      "host_latency_ms": ("Host TFLite latency per inference in ms",
                          SOFT_HOST_LATENCY_MS),
      "params_k": ("Weight parameters in thousands", SOFT_PARAMS_K),
      "mcu_ms_est": ("Estimated Cortex-M4F latency in ms (from MACs)",
                     SOFT_MCU_MS),
  }
  for name, (desc, budget) in lower.items():
    metric_lib.register_metric(metric_lib.Metric(
        name=name, func=_artifact_metric(name),
        description=f"{desc} (budget {budget:g})",
        higher_is_better=False, normalize=_budget_normalizer(budget),
    ))
  metric_lib.register_metric(metric_lib.Metric(
      name="int8_fraction", func=_artifact_metric("int8_fraction"),
      description="Fraction of dense/conv ops running on int8 activations",
      higher_is_better=True, normalize=None,
  ))
  metric_lib.register_metric(metric_lib.Metric(
      name="tflite_agreement", func=_artifact_metric("tflite_agreement"),
      description="Agreement between returned predictions and the TFLite"
                  " model run on raw X_val (quantization drift)",
      higher_is_better=True, normalize=None,
  ))
  metric_lib.register_metric(metric_lib.Metric(
      name="worst_class_recall", func=worst_class_recall,
      description="Recall of the worst class",
      higher_is_better=True, normalize=None,
  ))
  metric_lib.register_metric(metric_lib.Metric(
      name="null_fpr", func=null_fpr,
      description="False trigger rate: Null windows classified as a gesture",
      higher_is_better=False, normalize=lambda v: float(1.0 - min(1.0, v)),
  ))


def deploy_validator(
    fn_args: Sequence[Any],
    fn_kwargs: Mapping[str, Any],
    result: Any,
    num_classes: int | None = None,
    window_rows: int = data_loader.WINDOW_ROWS,
) -> None:
  """Validates predictions and the attached TFLite artifact.

  Raises:
    scoring.ValidatorError: On any contract violation. The message is fed
      back to the student as a traceback, so it is written as an instruction.
  """
  _tls.stats = None
  _tls.model_bytes = None
  scoring.classification_prediction_validator(
      fn_args, fn_kwargs, result, expected_columns=1
  )
  y_pred_df: pd.DataFrame = result[0]
  model_bytes = y_pred_df.attrs.pop(ATTR_KEY, None)
  if not isinstance(model_bytes, (bytes, bytearray)) or not model_bytes:
    raise scoring.ValidatorError(
        "Deploy contract: y_pred_df.attrs['tflite_model'] must hold the"
        " serialized TFLite model bytes (output of tf.lite.TFLiteConverter"
        " .convert()). Attach it before returning."
    )
  model_bytes = bytes(model_bytes)

  try:
    stats = analyze_tflite(model_bytes)
  except Exception as e:  # pylint: disable=broad-exception-caught
    raise scoring.ValidatorError(
        f"Deploy contract: TFLite model failed to load or run: {e}"
    ) from e

  n_features = window_rows * data_loader.NUM_AXES
  if int(np.prod(stats.input_shape)) != n_features:
    raise scoring.ValidatorError(
        f"Deploy contract: model input must hold exactly {n_features} values"
        f" (one {window_rows}x3 raw accelerometer window, batch 1); got"
        f" input shape {stats.input_shape}."
    )
  if num_classes is not None and stats.output_shape[-1] != num_classes:
    raise scoring.ValidatorError(
        f"Deploy contract: model output last dim must be {num_classes}"
        f" class scores; got {stats.output_shape}."
    )
  bad_ops = sorted(stats.ops - ALLOWED_OPS)
  if bad_ops:
    raise scoring.ValidatorError(
        f"Deploy contract: ops not supported by the firmware: {bad_ops}."
        f" Allowed: {sorted(ALLOWED_OPS)}."
    )
  if stats.model_kb > MAX_MODEL_KB:
    raise scoring.ValidatorError(
        f"Deploy contract: model is {stats.model_kb:.1f} KB, hard limit"
        f" {MAX_MODEL_KB:g} KB."
    )
  if stats.arena_kb > MAX_ARENA_KB:
    raise scoring.ValidatorError(
        f"Deploy contract: estimated arena {stats.arena_kb:.1f} KB exceeds"
        f" hard limit {MAX_ARENA_KB:g} KB."
    )

  x_val = fn_args[-1]
  features = data_loader.features_only(x_val)
  try:
    scores = run_tflite(model_bytes, features)
  except Exception as e:  # pylint: disable=broad-exception-caught
    raise scoring.ValidatorError(
        f"Deploy contract: running the TFLite model on X_val failed: {e}"
    ) from e
  tflite_pred = scores.argmax(axis=1)
  returned = y_pred_df.iloc[:, 0].to_numpy().astype(int)
  agreement = float(np.mean(tflite_pred == returned))
  stats.agreement = agreement
  if agreement < MIN_AGREEMENT:
    raise scoring.ValidatorError(
        "Deploy contract: predictions returned in y_pred_df must come from"
        f" the attached TFLite model run on raw X_val features; agreement was"
        f" only {agreement:.2%} (minimum {MIN_AGREEMENT:.0%}). Any"
        " normalization or feature extraction must live inside the TFLite"
        " graph, because the device feeds raw accelerometer values."
    )
  _tls.stats = stats
  _tls.model_bytes = model_bytes
  with _STORE_LOCK:
    if len(_ARTIFACTS) >= _ARTIFACT_LIMIT:
      _ARTIFACTS.pop(next(iter(_ARTIFACTS)))
    _ARTIFACTS[prediction_key(y_pred_df.index, returned)] = (model_bytes, stats)
  logger.info(
      "deploy ok: %.1f KB, %.3f MFLOPs, arena %.1f KB, %.2f ms, int8=%s,"
      " agreement %.2f, ops=%s",
      stats.model_kb, stats.mflops, stats.arena_kb, stats.host_latency_ms,
      stats.quantized, agreement, sorted(stats.ops),
  )


def write_model_header(
    path: str,
    model_bytes: bytes,
    names: Sequence[str],
    window_rows: int = data_loader.WINDOW_ROWS,
    extra_comment: str = "",
) -> None:
  """Writes `model_data.h` consumed by Arduino/Codegen."""
  lines = [
      "// Generated by Python/codegen_task; do not edit by hand.",
  ]
  if extra_comment:
    lines += [f"// {l}" for l in extra_comment.strip().splitlines()]
  lines += [
      "#pragma once",
      "",
      f"#define WINDOW_ROWS {window_rows}",
      f"#define NUM_AXES {data_loader.NUM_AXES}",
      f"#define NUM_CLASSES {len(names)}",
      "",
      "static const char* const GESTURES[NUM_CLASSES] = {",
      *[f'  "{n}",' for n in names],
      "};",
      "",
      "alignas(16) static const unsigned char g_model[] = {",
  ]
  for i in range(0, len(model_bytes), 12):
    chunk = model_bytes[i : i + 12]
    lines.append("  " + " ".join(f"0x{b:02x}," for b in chunk))
  lines += [
      "};",
      f"static const unsigned int g_model_len = {len(model_bytes)};",
      "",
  ]
  with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))


# ============================================================================
# On-device evaluation over the firmware REPLAY protocol
# ============================================================================

logger = logging.getLogger(__name__)


def find_port() -> str:
  ports = sorted(glob.glob("/dev/cu.usbmodem*"))
  if not ports:
    raise RuntimeError("No /dev/cu.usbmodem* port found")
  return ports[0]


def _read_until(ser: serial.Serial, prefix: str, timeout_s: float) -> str:
  t0 = time.time()
  while time.time() - t0 < timeout_s:
    line = ser.readline().decode(errors="ignore").strip()
    if line.startswith(prefix):
      return line
    if line.startswith("ERR:"):
      raise RuntimeError(line)
  raise TimeoutError(f"no {prefix} within {timeout_s}s")


def replay_window(ser: serial.Serial, window: np.ndarray) -> tuple[list[float], int]:
  """Sends one (rows, 3) window and returns (probabilities, latency_us)."""
  rows = window.shape[0]
  ser.reset_input_buffer()
  ser.write(f"REPLAY {rows}\n".encode())
  _read_until(ser, "READY", 3.0)
  for r in window:
    ser.write(f"{r[0]:.5f},{r[1]:.5f},{r[2]:.5f}\n".encode())
    _read_until(ser, ".", 2.0)  # per-row ack: the CDC RX buffer is small
  prob_line = _read_until(ser, "PROB:", 5.0)
  lat_line = _read_until(ser, "LAT:", 2.0)
  probs = [float(p) for p in prob_line[5:].split(",") if p.strip()]
  return probs, int(lat_line[4:])


def device_eval_main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--port", default=None)
  ap.add_argument("--split", default="test", choices=["test", "val", "all"])
  ap.add_argument("--json", default=None, help="Write results to this file")
  ap.add_argument("--data_dir", default=None)
  args = ap.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s")

  splits = data_loader.load_splits(args.data_dir)
  if args.split == "test":
    x, y = splits.x_test, splits.y_test
  elif args.split == "val":
    x, y = splits.x_val, splits.y_val
  else:
    x = pd.concat([splits.x_learning, splits.x_test]).reset_index(drop=True)
    y = pd.concat([splits.y_learning, splits.y_test]).reset_index(drop=True)
  feats = data_loader.features_only(x).reshape(len(x), -1, data_loader.NUM_AXES)
  y_true = y.iloc[:, 0].to_numpy(int)

  import serial  # pylint: disable=import-outside-toplevel
  port = args.port or find_port()
  with serial.Serial(port, 115200, timeout=0.5) as ser:
    time.sleep(0.5)
    ser.write(b"STATUS\n")
    info = _read_until(ser, "INFO:", 5.0)
    arena = int(_read_until(ser, "ARENA:", 5.0)[6:])
    logger.info("%s", info)
    logger.info("ARENA:%d", arena)
    preds, lats = [], []
    for i, w in enumerate(feats):
      probs, lat = replay_window(ser, w)
      preds.append(int(np.argmax(probs)))
      lats.append(lat)
      logger.info("%3d true=%s pred=%s lat=%dus", i,
                  splits.class_names[y_true[i]], splits.class_names[preds[-1]],
                  lat)
    ser.write(b"LIVE\n")

  from sklearn import metrics as skm  # pylint: disable=import-outside-toplevel
  preds_arr = np.asarray(preds)
  result = {
      "port": port,
      "split": args.split,
      "n": int(len(y_true)),
      "device_accuracy": float(np.mean(preds_arr == y_true)),
      "device_f1_macro": float(skm.f1_score(y_true, preds_arr, average="macro")),
      "device_latency_ms_mean": float(np.mean(lats) / 1000.0),
      "device_latency_ms_max": float(np.max(lats) / 1000.0),
      "arena_bytes": arena,
      "info": info,
      "confusion": skm.confusion_matrix(
          y_true, preds_arr, labels=range(len(splits.class_names))
      ).tolist(),
      "class_names": splits.class_names,
  }
  logger.info("device f1_macro=%.3f acc=%.3f latency=%.1f ms arena=%d B",
              result["device_f1_macro"], result["device_accuracy"],
              result["device_latency_ms_mean"], arena)
  import json  # pylint: disable=import-outside-toplevel
  import os  # pylint: disable=import-outside-toplevel
  repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  eval_dir = os.path.join(repo_root, "experiments", "device_eval")
  os.makedirs(eval_dir, exist_ok=True)
  ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
  ts_path = os.path.join(eval_dir, f"device_eval_{ts}.json")
  with open(ts_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
  out_path = args.json or os.path.join(eval_dir, "device_eval_latest.json")
  os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
  with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
  logger.info("Saved device eval to %s and %s", out_path, ts_path)


register_metrics()


if __name__ == "__main__":
  device_eval_main()
