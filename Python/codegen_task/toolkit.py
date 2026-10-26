"""Shared toolkit students may import inside `solution()`.

    from codegen_task import toolkit as tk

Covers the plumbing that costs classroom attempts: tensor shaping, realistic
augmentation, hard negatives, in-graph normalization, class weights,
full-int8 conversion, running the converted model, budget inspection, and
building the return value. Modeling decisions stay with the student.
Everything here is deterministic given `seed`.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from codegen_task import data_loader
from codegen_task import deploy

GROUP_COLUMN = data_loader.GROUP_COLUMN
NUM_AXES = data_loader.NUM_AXES


def set_determinism(seed: int = 0) -> None:
  """Seeds numpy, random, and TensorFlow and enables deterministic ops."""
  import tensorflow as tf  # pylint: disable=import-outside-toplevel
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  try:
    tf.get_logger().setLevel("ERROR")
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  try:
    if hasattr(tf.keras.optimizers, "legacy"):
      tf.keras.optimizers.Adam = tf.keras.optimizers.legacy.Adam
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  try:
    tf.config.experimental.enable_op_determinism()
  except Exception:  # pylint: disable=broad-exception-caught
    pass


def null_index() -> int | None:
  """Class index of `Null` for this dataset, or None if there is no Null."""
  return deploy._null_index  # pylint: disable=protected-access


def split_xy(
    x_train: pd.DataFrame, y_train: pd.DataFrame, x_val: pd.DataFrame
) -> dict[str, Any]:
  """Returns a dict of arrays shaped for sequence models (not a tuple).

  Keys: `x_tr` (n, rows, 3) float32, `y_tr` int, `x_va` (m, rows, 3),
  `groups` recording ids for `x_tr`, `rows`, `n_classes` (from y_train).
  Usage: `d = tk.split_xy(X_train, y_train, X_val); x = d["x_tr"]`.
  """
  feat = [c for c in x_train.columns if c != GROUP_COLUMN]
  rows = len(feat) // NUM_AXES
  x_tr = x_train[feat].to_numpy(np.float32).reshape(-1, rows, NUM_AXES)
  x_va = x_val[feat].to_numpy(np.float32).reshape(-1, rows, NUM_AXES)
  y_tr = y_train.iloc[:, 0].to_numpy(int)
  groups = (x_train[GROUP_COLUMN].to_numpy()
            if GROUP_COLUMN in x_train else np.arange(len(x_train)))
  return {"x_tr": x_tr, "y_tr": y_tr, "x_va": x_va, "groups": groups,
          "rows": rows, "n_classes": int(y_tr.max()) + 1}


def _rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
  a = np.deg2rad(rng.uniform(-max_deg, max_deg, size=3))
  cx, cy, cz = np.cos(a)
  sx, sy, sz = np.sin(a)
  rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
  ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
  rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
  return (rz @ ry @ rx).astype(np.float32)


def _stretch(w: np.ndarray, factor: float) -> np.ndarray:
  rows = w.shape[0]
  src = np.linspace(0, rows - 1, int(round(rows * factor)))
  out = np.stack([np.interp(src, np.arange(rows), w[:, a])
                  for a in range(w.shape[1])], axis=1)
  if len(out) >= rows:
    s = (len(out) - rows) // 2
    return out[s:s + rows]
  pad = rows - len(out)
  return np.pad(out, ((pad // 2, pad - pad // 2), (0, 0)), mode="edge")


def _edge_shift(w: np.ndarray, shift: int) -> np.ndarray:
  if shift > 0:
    return np.pad(w[:-shift], ((shift, 0), (0, 0)), mode="edge")
  if shift < 0:
    return np.pad(w[-shift:], ((0, -shift), (0, 0)), mode="edge")
  return w.copy()


def augment(
    x: np.ndarray, y: np.ndarray, reps: int, seed: int = 0, *,
    scale: tuple[float, float] = (0.8, 1.2),
    shift_frac: float = 0.15,
    rotate_deg: float = 10.0,
    stretch: tuple[float, float] = (0.85, 1.18),
    noise: float = 0.02,
    blend_prob: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
  """Physically plausible augmentation of (n, rows, 3) windows.

  Separates static gravity pose from dynamic impulse so dynamic amplitude
  scales by `scale` while resting gravity stays near 1 g, optionally blends
  peak-aligned same-class takes, then applies a small 3D rotation, time
  stretching with resampling, edge-padded time shift, and sensor noise.
  Returns the originals followed by `reps` copies.
  """
  rng = np.random.default_rng(seed)
  rows = x.shape[1]
  by_cls: dict[int, np.ndarray] = {
      int(c): np.where(y == c)[0] for c in np.unique(y)
  }
  xs, ys = [x.astype(np.float32)], [y]
  for _ in range(reps):
    out = np.empty_like(x, dtype=np.float32)
    for i, w in enumerate(x):
      pose = w.mean(axis=0, keepdims=True)
      dyn = w - pose
      peers = by_cls.get(int(y[i]))
      if peers is not None and len(peers) > 1 and rng.random() < blend_prob:
        w2 = x[int(rng.choice(peers))]
        pose2 = w2.mean(axis=0, keepdims=True)
        dyn2 = w2 - pose2
        p1 = int(np.argmax(np.sum(dyn * dyn, axis=-1)))
        p2 = int(np.argmax(np.sum(dyn2 * dyn2, axis=-1)))
        dyn2 = _edge_shift(dyn2, int(np.clip(p1 - p2, -16, 16)))
        alpha = float(rng.uniform(0.2, 0.45))
        pose = (1.0 - alpha) * pose + alpha * pose2
        dyn = (1.0 - alpha) * dyn + alpha * dyn2
      v = pose * rng.uniform(0.96, 1.04) + dyn * rng.uniform(*scale)
      v = v @ _rotation(rng, rotate_deg).T
      v = _stretch(v, rng.uniform(*stretch))
      max_sh = max(1, int(rows * shift_frac))
      v = _edge_shift(v, int(rng.integers(-max_sh, max_sh + 1)))
      out[i] = v + rng.normal(0, noise, size=v.shape)
    xs.append(out)
    ys.append(y)
  return np.concatenate(xs), np.concatenate(ys)


def hard_negatives(
    x: np.ndarray, y: np.ndarray, null_index: int | None = None,
    n: int = 32, seed: int = 0, **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
  """Synthesizes `n` realistic Null handling windows the device sees at rest.

  Mixes attenuated boundary gesture tails/onsets (<18% window at low dynamic
  amplitude), tilted ~1 g static holds with slow wrist posture drift,
  low-energy idle segments from real takes, and mild desk/cable handling bumps.
  """
  if null_index is None:
    null_index = kwargs.get("null_idx", deploy._null_index)  # pylint: disable=protected-access
  if null_index is None:
    null_index = 2
  null_index = int(null_index)
  rng = np.random.default_rng(seed)
  rows = x.shape[1]
  out = np.empty((n, rows, NUM_AXES), np.float32)
  gest = x[y != null_index] if (y != null_index).any() else x
  for i in range(n):
    kind = i % 4
    w = gest[rng.integers(len(gest))].copy()
    if kind == 0:  # attenuated boundary gesture onset/tail + still hand tremor
      cut = int(rng.integers(max(4, rows // 10), max(6, rows // 5)))
      pose = w.mean(axis=0, keepdims=True)
      w = pose + (w - pose) * float(rng.uniform(0.08, 0.20))
      if rng.random() < 0.5:
        still = w[cut - 1:cut]
        w[cut:] = still + rng.normal(0, 0.015, size=(rows - cut, NUM_AXES))
      else:
        still = w[rows - cut:rows - cut + 1]
        w[:rows - cut] = still + rng.normal(
            0, 0.015, size=(rows - cut, NUM_AXES)
        )
    elif kind == 1:  # tilted ~1 g pose with slow wrist drift and tremor
      g0 = _rotation(rng, 180.0) @ np.array([0, 0, 1.0], np.float32)
      g1 = g0 + rng.normal(0, 0.06, size=NUM_AXES).astype(np.float32)
      g1 = g1 / (np.linalg.norm(g1) + 1e-6) * float(rng.uniform(0.96, 1.04))
      alpha = np.linspace(0.0, 1.0, rows, dtype=np.float32)[:, None]
      w = (1.0 - alpha) * g0[None, :] + alpha * g1[None, :]
      w = w + rng.normal(0, rng.uniform(0.01, 0.025), size=(rows, NUM_AXES))
    elif kind == 2:  # quietest sub-window from a real recording clamped to low energy
      seg_len = min(20, rows)
      starts = range(0, max(1, rows - seg_len + 1), max(1, seg_len // 2))
      s0 = min(starts, key=lambda s: float(np.std(w[s:s + seg_len], axis=0).sum()))
      quiet = w[s0:s0 + seg_len].copy()
      q_pose = quiet.mean(axis=0, keepdims=True)
      q_dyn = quiet - q_pose
      max_d = float(np.max(np.linalg.norm(q_dyn, axis=-1)) + 1e-6)
      if max_d > 0.25:
        q_dyn = q_dyn * (0.20 / max_d)
      quiet = q_pose + q_dyn
      reps_q = (rows + seg_len - 1) // seg_len
      w = np.tile(quiet, (reps_q, 1))[:rows]
      w = w + rng.normal(0, 0.015, size=(rows, NUM_AXES))
    else:  # low-amplitude desk bump / cable tug on a resting ~1 g pose
      g0 = _rotation(rng, 180.0) @ np.array([0, 0, 1.0], np.float32)
      g0 = g0 * float(rng.uniform(0.97, 1.03))
      w = np.broadcast_to(g0[None, :], (rows, NUM_AXES)).copy()
      w = w + rng.normal(0, 0.015, size=(rows, NUM_AXES)).astype(np.float32)
      dur = min(int(rng.integers(6, 14)), max(2, rows // 4))
      t0 = int(rng.integers(0, max(1, rows - dur)))
      env = np.hanning(dur)[:, None].astype(np.float32)
      bump = rng.normal(0, float(rng.uniform(0.08, 0.20)), size=(1, NUM_AXES))
      w[t0:t0 + dur] += env * bump.astype(np.float32)
    out[i] = w.astype(np.float32)
  return out, np.full(n, null_index, int)


def norm_layer(x_tr: np.ndarray):
  """Keras layer that standardizes per axis with constants from `x_tr`.

  Converts to MUL/SUB (or folds into the next layer), so the device can
  feed raw g values. Apply as the first layer after the Input.
  """
  import tensorflow as tf  # pylint: disable=import-outside-toplevel
  flat = x_tr.reshape(-1, x_tr.shape[-1])
  mean = flat.mean(0).astype(np.float32)
  std = (flat.std(0) + 1e-6).astype(np.float32)
  return tf.keras.layers.Lambda(lambda t: (t - mean) / std, name="norm")


def class_weights(y: np.ndarray, n_classes: int) -> dict[int, float]:
  """Balanced class weights with a key for every class index."""
  counts = np.bincount(y, minlength=n_classes).astype(float)
  counts[counts == 0] = counts[counts > 0].mean()
  w = counts.sum() / (n_classes * counts)
  return {i: float(w[i]) for i in range(n_classes)}


def to_int8_tflite(
    model, rep_x: np.ndarray, io_int8: bool = True, max_rep: int = 300
) -> bytes:
  """Full-integer int8 conversion with `rep_x` as representative data."""
  import tensorflow as tf  # pylint: disable=import-outside-toplevel
  rep = rep_x.astype(np.float32)[:max_rep]

  def gen():
    for i in range(len(rep)):
      yield [rep[i:i + 1]]

  conv = tf.lite.TFLiteConverter.from_keras_model(model)
  conv.optimizations = [tf.lite.Optimize.DEFAULT]
  conv.representative_dataset = gen
  conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  if io_int8:
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
  return conv.convert()


def to_float_tflite(model) -> bytes:
  """Plain float32 conversion (larger and slower on the device)."""
  import tensorflow as tf  # pylint: disable=import-outside-toplevel
  return tf.lite.TFLiteConverter.from_keras_model(model).convert()


def predict_tflite(tflite: bytes, x: np.ndarray) -> np.ndarray:
  """Class scores (n, n_classes) from the converted model on raw windows."""
  return deploy.run_tflite(tflite, x.reshape(len(x), -1))


def inspect(tflite: bytes) -> dict[str, Any]:
  """Size, MACs, arena estimate, ops, and int8 status for a flatbuffer."""
  st = deploy.analyze_tflite(tflite, latency_runs=5)
  return {"model_kb": st.model_kb, "size_kb": st.model_kb,
          "mflops": st.mflops, "arena_kb": st.arena_kb,
          "params_k": st.params_k, "mcu_ms_est": st.mcu_ms_est,
          "int8_fraction": st.int8_fraction, "ops": sorted(st.ops),
          "input_shape": st.input_shape}


def check_contract(tflite: bytes, n_classes: int) -> list[str]:
  """Lists deploy-contract violations (empty list means deployable)."""
  st = deploy.analyze_tflite(tflite, latency_runs=1)
  problems = []
  need = data_loader.WINDOW_ROWS * NUM_AXES
  if int(np.prod(st.input_shape)) != need:
    problems.append(f"input holds {int(np.prod(st.input_shape))} values,"
                    f" need {need}")
  if st.output_shape[-1] != n_classes:
    problems.append(f"output has {st.output_shape[-1]} scores, need"
                    f" {n_classes}")
  bad = sorted(st.ops - deploy.ALLOWED_OPS)
  if bad:
    problems.append(f"unsupported ops {bad}")
  if st.model_kb > deploy.MAX_MODEL_KB:
    problems.append(f"{st.model_kb:.1f} KB > {deploy.MAX_MODEL_KB:g} KB")
  if st.arena_kb > deploy.MAX_ARENA_KB:
    problems.append(f"arena {st.arena_kb:.1f} KB > {deploy.MAX_ARENA_KB:g}")
  return problems


def grouped_oof(
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray, y: np.ndarray, groups: Sequence[Any], k: int = 3,
) -> tuple[np.ndarray, float]:
  """Grouped out-of-fold predictions and macro F1 using `fit_predict`.

  `fit_predict(x_fit, y_fit, x_pred)` must return class indices. `k` is
  capped by the smallest class count so folds are always valid.
  """
  from sklearn import metrics as skm  # pylint: disable=import-outside-toplevel
  from sklearn import model_selection  # pylint: disable=import-outside-toplevel
  k = int(max(2, min(k, np.bincount(y).min())))
  oof = np.zeros(len(y), int)
  for tr, te in model_selection.StratifiedGroupKFold(
      n_splits=k, shuffle=True, random_state=0).split(x, y, groups):
    oof[te] = fit_predict(x[tr], y[tr], x[te])
  return oof, float(skm.f1_score(y, oof, average="macro", zero_division=0))


def make_result(
    x_val: pd.DataFrame, y_train: pd.DataFrame, preds: np.ndarray,
    tflite: bytes, feedback: str,
) -> tuple[pd.DataFrame, str]:
  """Builds the `(y_pred_df, feedback)` return with the model attached."""
  df = pd.DataFrame({y_train.columns[0]: np.asarray(preds, int)},
                    index=x_val.index)
  df.attrs[deploy.ATTR_KEY] = bytes(tflite)
  return df, feedback


def api_summary() -> str:
  """Signature plus first docstring paragraph per helper, for the prompt."""
  import inspect as _inspect  # pylint: disable=import-outside-toplevel
  names = ("set_determinism", "split_xy", "null_index", "augment",
           "hard_negatives", "norm_layer", "class_weights", "to_int8_tflite",
           "to_float_tflite", "predict_tflite", "inspect", "check_contract",
           "grouped_oof", "make_result")
  lines = []
  for n in names:
    fn = globals()[n]
    sig = str(_inspect.signature(fn))
    sig = sig.replace("numpy.ndarray", "ndarray").replace(
        "pandas.core.frame.DataFrame", "DataFrame")
    doc = (fn.__doc__ or "").strip()
    para = " ".join(l.strip() for l in doc.split("\n\n")[0].splitlines())
    if len(doc.split("\n\n")) > 1 and n in ("split_xy", "grouped_oof"):
      para += " " + " ".join(
          l.strip() for l in doc.split("\n\n")[1].splitlines())
    lines.append(f"- `tk.{n}{sig}`: {para}")
  return "\n".join(lines)
