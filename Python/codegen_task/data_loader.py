"""Dataset loader for the micro_gesture classroom task.

Reads the playground's raw recordings (`Python/data/<Label>-<n>.txt`, one
`ax ay az` row per 10 ms sample), turns each recording into a fixed-size
window in firmware order, and assigns recordings to train / val / test with a
deterministic per-class hash so the held-out test set stays frozen when new
recordings arrive.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import re
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WINDOW_ROWS = 100
NUM_AXES = 3
AXES = ("x", "y", "z")
TARGET_COLUMN = "label"
GROUP_COLUMN = "recording_id"
FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"s{r:03d}_{a}" for r in range(WINDOW_ROWS) for a in AXES
)
_FILE_RE = re.compile(r"^(?P<label>[A-Za-z][A-Za-z0-9_]*)-(?P<idx>\d+)$")

# Fraction of each class routed to the frozen test split and to validation.
TEST_FRACTION = 0.25
VAL_FRACTION = 0.20
EXCLUDED_LABELS: frozenset[str] = frozenset({"Drop"})


def default_data_dir() -> str:
  """Returns `PLAYGROUND_DATA_DIR` or the sibling `Python/data` directory."""
  env = os.environ.get("PLAYGROUND_DATA_DIR")
  if env:
    return os.path.abspath(env)
  here = os.path.dirname(os.path.abspath(__file__))
  return os.path.normpath(os.path.join(here, "..", "data"))


@dataclasses.dataclass(frozen=True)
class Recording:
  """One labelled accelerometer recording."""

  recording_id: str
  label: str
  samples: np.ndarray  # (rows, 3) float32


def _stable_hash(text: str) -> int:
  return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)


def list_recordings(data_dir: str) -> list[Recording]:
  """Parses every `<Label>-<n>.txt` file in `data_dir`."""
  recs: list[Recording] = []
  for name in sorted(os.listdir(data_dir)):
    stem, ext = os.path.splitext(name)
    if ext != ".txt":
      continue
    m = _FILE_RE.match(stem)
    if not m:
      logger.warning("Skipping unrecognised file name: %s", name)
      continue
    label = m.group("label")
    if label in EXCLUDED_LABELS:
      continue
    arr = np.loadtxt(os.path.join(data_dir, name), dtype=np.float32)
    if arr.ndim == 1:
      arr = arr.reshape(-1, NUM_AXES)
    if arr.shape[1] != NUM_AXES or arr.shape[0] < 10:
      logger.warning("Skipping malformed recording %s: %s", name, arr.shape)
      continue
    recs.append(Recording(stem, label, arr))
  if not recs:
    raise FileNotFoundError(f"No recordings found in {data_dir}")
  return recs


def window_recording(
    samples: np.ndarray, window_rows: int = WINDOW_ROWS
) -> np.ndarray:
  """Center-crops or edge-pads a recording to exactly `window_rows` rows."""
  n = samples.shape[0]
  if n >= window_rows:
    start = (n - window_rows) // 2
    return samples[start : start + window_rows]
  pad = window_rows - n
  before = pad // 2
  return np.pad(samples, ((before, pad - before), (0, 0)), mode="edge")


def dataset_fingerprint(data_dir: str) -> str:
  """Hash of recording names and sizes, used to detect new data."""
  h = hashlib.sha1()
  for name in sorted(os.listdir(data_dir)):
    if not name.endswith(".txt"):
      continue
    stem = os.path.splitext(name)[0]
    m = _FILE_RE.match(stem)
    if m and m.group("label") in EXCLUDED_LABELS:
      continue
    size = os.path.getsize(os.path.join(data_dir, name))
    h.update(f"{name}:{size}\n".encode("utf-8"))
  return h.hexdigest()[:12]


def class_names(recordings: Sequence[Recording]) -> list[str]:
  """Sorted class names; the index in this list is the integer label."""
  return sorted({r.label for r in recordings})


def build_frame(
    recordings: Sequence[Recording],
    names: Sequence[str],
    window_rows: int = WINDOW_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Builds (X, y) with one row per recording in firmware feature order."""
  rows = [window_recording(r.samples, window_rows).reshape(-1) for r in recordings]
  cols = [f"s{r:03d}_{a}" for r in range(window_rows) for a in AXES]
  x = pd.DataFrame(np.stack(rows), columns=cols)
  x[GROUP_COLUMN] = [r.recording_id for r in recordings]
  label_idx = {n: i for i, n in enumerate(names)}
  y = pd.DataFrame({TARGET_COLUMN: [label_idx[r.label] for r in recordings]})
  return x, y


def assign_split(recordings: Sequence[Recording]) -> dict[str, str]:
  """Maps recording_id -> 'train' | 'val' | 'test' by per-class hash rank.

  Within each class, recordings are ordered by a stable hash of their id; the
  lowest `TEST_FRACTION` become the frozen test set, the next `VAL_FRACTION`
  validation. Adding recordings only shifts boundaries slightly, and a given
  recording never moves from test back into the learning splits.
  """
  split: dict[str, str] = {}
  by_label: dict[str, list[Recording]] = {}
  for r in recordings:
    by_label.setdefault(r.label, []).append(r)
  for label, recs in by_label.items():
    ordered = sorted(recs, key=lambda r: _stable_hash(r.recording_id))
    n = len(ordered)
    n_test = max(1, int(round(n * TEST_FRACTION))) if n >= 3 else 0
    n_val = max(1, int(round(n * VAL_FRACTION))) if n - n_test >= 2 else 0
    for i, r in enumerate(ordered):
      if i < n_test:
        split[r.recording_id] = "test"
      elif i < n_test + n_val:
        split[r.recording_id] = "val"
      else:
        split[r.recording_id] = "train"
    logger.info(
        "class %-20s n=%3d train=%3d val=%3d test=%3d",
        label, n, n - n_test - n_val, n_val, n_test,
    )
  return split


@dataclasses.dataclass
class Splits:
  """Train / val / test frames plus metadata."""

  x_train: pd.DataFrame
  y_train: pd.DataFrame
  x_val: pd.DataFrame
  y_val: pd.DataFrame
  x_test: pd.DataFrame
  y_test: pd.DataFrame
  class_names: list[str]
  fingerprint: str
  data_dir: str = ""

  @property
  def x_learning(self) -> pd.DataFrame:
    return pd.concat([self.x_train, self.x_val]).reset_index(drop=True)

  @property
  def y_learning(self) -> pd.DataFrame:
    return pd.concat([self.y_train, self.y_val]).reset_index(drop=True)

  @property
  def null_index(self) -> int | None:
    return self.class_names.index("Null") if "Null" in self.class_names else None

  @property
  def x_val_eval(self) -> pd.DataFrame:
    return expand_eval_frame(
        self.x_val, self.y_val, self.null_index, seed=88
    )[0]

  @property
  def y_val_eval(self) -> pd.DataFrame:
    return expand_eval_frame(
        self.x_val, self.y_val, self.null_index, seed=88
    )[1]

  @property
  def x_test_eval(self) -> pd.DataFrame:
    return expand_eval_frame(
        self.x_test, self.y_test, self.null_index, seed=99
    )[0]

  @property
  def y_test_eval(self) -> pd.DataFrame:
    return expand_eval_frame(
        self.x_test, self.y_test, self.null_index, seed=99
    )[1]


def expand_eval_frame(
    x_df: pd.DataFrame,
    y_df: pd.DataFrame,
    null_index: int | None = None,
    seed: int = 99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Expands a held-out split with deterministic on-device streaming & Null views.

  Each held-out recording produces:
  1. Its canonical center-cropped window (t=0).
  2. An early-arrival sliding window (-14..-22 samples, +/-9 deg tilt, scale/stretch).
  3. A late-arrival sliding window (+14..+22 samples, +/-9 deg tilt, scale/stretch).
  Plus 4 realistic handling Null boundary windows per Null recording in the split,
  all generated strictly from `x_df` so zero cross-split leakage can occur.
  """
  from codegen_task import toolkit as tk  # pylint: disable=import-outside-toplevel
  feat_cols = [c for c in x_df.columns if c != GROUP_COLUMN]
  rows = len(feat_cols) // NUM_AXES
  x_arr = x_df[feat_cols].to_numpy(np.float32).reshape(-1, rows, NUM_AXES)
  y_arr = y_df[TARGET_COLUMN].to_numpy(int)
  rids = (
      x_df[GROUP_COLUMN].to_numpy(str)
      if GROUP_COLUMN in x_df.columns
      else np.array([f"r{i}" for i in range(len(x_df))])
  )
  rng = np.random.default_rng(seed)
  views_x = [x_arr]
  views_y = [y_arr]
  views_id = [rids]
  max_sh = max(2, int(round(rows * 0.20)))
  min_sh = max(1, int(round(rows * 0.13)))
  for sign in (-1, 1):
    v_arr = np.empty_like(x_arr)
    for i, w in enumerate(x_arr):
      pose = w.mean(axis=0, keepdims=True)
      dyn = w - pose
      v = pose * rng.uniform(0.97, 1.03) + dyn * rng.uniform(0.84, 1.18)
      v = v @ tk._rotation(rng, 9.0).T  # pylint: disable=protected-access
      v = tk._stretch(v, rng.uniform(0.88, 1.14))  # pylint: disable=protected-access
      sh = sign * int(rng.integers(min_sh, max_sh + 1))
      v = tk._edge_shift(v, sh)  # pylint: disable=protected-access
      v_arr[i] = v + rng.normal(0, 0.018, size=v.shape)
    views_x.append(v_arr.astype(np.float32))
    views_y.append(y_arr)
    views_id.append(rids)
  if null_index is not None and len(x_arr) >= 4:
    n_null_recs = max(1, int(np.sum(y_arr == null_index)))
    n_hn = 4 * n_null_recs
    hn_x, hn_y = tk.hard_negatives(
        x_arr, y_arr, null_index=null_index, n=n_hn, seed=seed + 17
    )
    null_rids = rids[y_arr == null_index]
    if len(null_rids) == 0:
      null_rids = rids[:1]
    hn_ids = np.array([null_rids[i % len(null_rids)] for i in range(n_hn)])
    views_x.append(hn_x)
    views_y.append(hn_y)
    views_id.append(hn_ids)
  all_x = np.concatenate(views_x, axis=0).reshape(-1, rows * NUM_AXES)
  all_y = np.concatenate(views_y, axis=0)
  all_ids = np.concatenate(views_id, axis=0)
  out_x = pd.DataFrame(all_x, columns=feat_cols)
  out_x[GROUP_COLUMN] = all_ids
  out_y = pd.DataFrame({TARGET_COLUMN: all_y})
  return out_x, out_y


def load_splits(
    data_dir: str | None = None, window_rows: int = WINDOW_ROWS
) -> Splits:
  """Loads all recordings and returns frozen train / val / test splits."""
  data_dir = data_dir or default_data_dir()
  recs = list_recordings(data_dir)
  names = class_names(recs)
  split = assign_split(recs)
  parts = {k: [r for r in recs if split[r.recording_id] == k]
           for k in ("train", "val", "test")}
  frames = {k: build_frame(v, names, window_rows) for k, v in parts.items()}
  return Splits(
      x_train=frames["train"][0], y_train=frames["train"][1],
      x_val=frames["val"][0], y_val=frames["val"][1],
      x_test=frames["test"][0], y_test=frames["test"][1],
      class_names=names,
      fingerprint=dataset_fingerprint(data_dir),
      data_dir=data_dir,
  )


def features_only(x: pd.DataFrame) -> np.ndarray:
  """Drops the group column and returns a float32 (n, rows*3) matrix."""
  return x.drop(columns=[GROUP_COLUMN], errors="ignore").to_numpy(np.float32)
