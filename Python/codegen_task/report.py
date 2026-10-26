"""Solution report for a micro_gesture run: dataset, model, benchmarks, HIL.

Builds `export/REPORT.md` with figures under `export/figs/` from the winning
artifact of a run. Sections degrade gracefully: any failing section is noted
and the rest of the report is still written.

Usage (standalone, regenerates for an existing run):
  python -m codegen_task.report --run_dir latest [--hil auto|on|off]
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from codegen_task import data_loader
from codegen_task import deploy

logger = logging.getLogger(__name__)

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.normpath(os.path.join(TASK_DIR, "..", ".."))
FIRMWARE_HEADER = os.path.join(REPO_DIR, "Arduino", "Codegen", "model_data.h")


# Figure conventions follow codegen_classroom.plotting: tab10 palette, bold
# titles, dashed grids, 150 dpi, 4.8 x 4.2 in per panel, panel letters.
PANEL_W, PANEL_H, DPI = 4.8, 4.2, 150
_PALETTE = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3"]
_fig_counter = {"n": 0}


def _plt():
  import matplotlib  # pylint: disable=import-outside-toplevel
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
  plt.style.use("default")
  plt.rcParams.update({
      "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
      "axes.labelsize": 10, "legend.fontsize": 8, "xtick.labelsize": 9,
      "ytick.labelsize": 9, "axes.spines.top": False,
      "axes.spines.right": False, "legend.frameon": False,
      "axes.prop_cycle": matplotlib.cycler(
          color=plt.get_cmap("tab10").colors),
  })
  return plt


def _subplots(ncols: int = 1, nrows: int = 1, w: float = PANEL_W,
              h: float = PANEL_H, **kw):
  plt = _plt()
  fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows),
                           squeeze=False, **kw)
  return fig, axes


def _finish(fig, path: str, letters: bool = True) -> None:
  """Grid, panel letters, tight layout, save at print dpi."""
  axes = [a for a in fig.axes if a.get_visible() and a.get_label() != "cbar"]
  for i, ax in enumerate(axes):
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    if letters and len(axes) > 1:
      ax.text(-0.08, 1.06, f"({chr(97 + i)})", transform=ax.transAxes,
              fontsize=11, fontweight="bold", va="bottom", ha="right")
  fig.tight_layout()
  fig.savefig(path, dpi=DPI, bbox_inches="tight")
  import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
  plt.close(fig)


def _caption(name: str, text: str) -> str:
  """Image plus a numbered italic caption."""
  _fig_counter["n"] += 1
  return f"![{name}](figs/{name}.png)\n\n*Figure {_fig_counter['n']}. {text}*"


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
  out = ["| " + " | ".join(headers) + " |",
         "| " + " | ".join(":---" for _ in headers) + " |"]
  for r in rows:
    out.append("| " + " | ".join(_fmt(v) for v in r) + " |")
  return "\n".join(out)


def _fmt(v: Any) -> str:
  if isinstance(v, float):
    return f"{v:.4g}"
  return str(v)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def dataset_section(splits: data_loader.Splits, figs: str) -> str:
  recs = data_loader.list_recordings(
      splits.data_dir or data_loader.default_data_dir())
  names = splits.class_names
  tgt = data_loader.TARGET_COLUMN
  rows = []
  kin_rows = []
  feat_mat = []
  all_pts = []
  for i, n in enumerate(names):
    cls_recs = [r for r in recs if r.label == n]
    w_cls = np.stack([data_loader.window_recording(r.samples)
                      for r in cls_recs])  # (N_c, 100, 3)
    pose_mean = w_cls.mean(axis=1)  # (N_c, 3)
    dyn = w_cls - w_cls.mean(axis=1, keepdims=True)
    peak_dyn = np.linalg.norm(dyn, axis=2).max(axis=1)  # (N_c,)
    jerk = np.diff(w_cls, axis=1) * 100.0
    rms_jerk = np.sqrt(np.mean(np.sum(jerk ** 2, axis=2), axis=1))
    # Dominant frequency from FFT of dynamic magnitude
    dyn_mag = np.linalg.norm(dyn, axis=2)
    fft_mag = np.abs(np.fft.rfft(dyn_mag - dyn_mag.mean(axis=1, keepdims=True),
                                 axis=1))
    freqs = np.fft.rfftfreq(w_cls.shape[1], d=0.01)
    dom_hz = freqs[np.argmax(fft_mag[:, 1:], axis=1) + 1]
    for z_v, pk_v in zip(pose_mean[:, 2], peak_dyn):
      all_pts.append((i, n, float(z_v), float(pk_v)))
    mx, my, mz = pose_mean.mean(axis=0)
    pk_m = float(peak_dyn.mean())
    jk_m = float(rms_jerk.mean())
    hz_m = float(dom_hz.mean())
    feat_mat.append([mx, my, mz, pk_m, jk_m, hz_m])
    rows.append([
        n,
        int((splits.y_train[tgt] == i).sum()),
        int((splits.y_val[tgt] == i).sum()),
        int((splits.y_test[tgt] == i).sum()),
        len(cls_recs),
        f"({mx:+.2f}, {my:+.2f}, {mz:+.2f})",
        f"{pk_m:.2f}",
        f"{jk_m:.1f}",
        f"{hz_m:.1f}",
    ])
  rows.append(["total", len(splits.x_train), len(splits.x_val),
               len(splits.x_test), len(recs), "-", "-", "-", "-"])
  lengths = np.array([r.samples.shape[0] for r in recs])
  mags = np.concatenate([np.linalg.norm(r.samples, axis=1) for r in recs])

  fig, axes = _subplots(2, 1, h=3.4)
  ax = axes[0][0]
  ax.hist(lengths, bins=20, color=_PALETTE[0], edgecolor="white")
  ax.axvline(data_loader.WINDOW_ROWS, color="k", ls="--", lw=1,
             label=f"window = {data_loader.WINDOW_ROWS} rows")
  ax.set_xlabel("recording length (rows at 100 Hz)")
  ax.set_ylabel("recordings")
  ax.set_title("Recording length")
  ax.legend()
  ax = axes[0][1]
  ax.hist(mags, bins=60, color=_PALETTE[2], edgecolor="white")
  ax.axvline(1.0, color="k", ls="--", lw=1, label="1 g")
  ax.set_xlabel("acceleration magnitude |a| (g)")
  ax.set_ylabel("samples")
  ax.set_title("Magnitude distribution")
  ax.legend()
  _finish(fig, os.path.join(figs, "dataset_hist.png"))

  # Kinematic separability figure: (a) Pose Z vs Peak Dynamic |a|, (b) Feature heatmap
  plt = _plt()
  cmap = plt.get_cmap("tab10")
  fig, axes_k = _subplots(2, 1, w=5.2, h=4.2)
  ax0 = axes_k[0][0]
  for i, n in enumerate(names):
    pts_c = [(z_v, pk_v) for ci, _, z_v, pk_v in all_pts if ci == i]
    if pts_c:
      ax0.scatter([p[0] for p in pts_c], [p[1] for p in pts_c],
                  color=cmap(i % 10), s=28, alpha=0.85, label=n,
                  edgecolors="white", lw=0.4)
  ax0.axvline(0.0, color="k", ls=":", lw=0.8)
  ax0.set_xlabel("mean Z-axis gravity pose (g)")
  ax0.set_ylabel("peak dynamic impulse |a - mean| (g)")
  ax0.set_title("Physical pose vs. dynamic impulse space")
  ax0.legend(loc="upper left", ncol=2, fontsize=7.5)

  ax1 = axes_k[0][1]
  f_arr = np.asarray(feat_mat, dtype=float)
  f_z = (f_arr - f_arr.mean(axis=0, keepdims=True)) / (
      f_arr.std(axis=0, keepdims=True) + 1e-6)
  im = ax1.imshow(f_z, cmap="coolwarm", vmin=-2.2, vmax=2.2, aspect="auto")
  f_labels = ["Mean X", "Mean Y", "Mean Z", "Peak |a_dyn|", "RMS Jerk", "Freq Hz"]
  ax1.set_xticks(range(len(f_labels)))
  ax1.set_xticklabels(f_labels, rotation=30, ha="right", fontsize=8)
  ax1.set_yticks(range(len(names)))
  ax1.set_yticklabels(names, fontsize=8)
  for r_i in range(len(names)):
    for c_i in range(len(f_labels)):
      val_s = f"{f_arr[r_i, c_i]:.1f}" if c_i >= 3 else f"{f_arr[r_i, c_i]:+.2f}"
      ax1.text(c_i, r_i, val_s, ha="center", va="center", fontsize=7.5,
               color="white" if abs(f_z[r_i, c_i]) > 1.2 else "black")
  ax1.set_title("Per-class physical signature (raw values, Z-score color)")
  ax1.grid(False)
  _finish(fig, os.path.join(figs, "kinematic_space.png"))
  del kin_rows

  cols = 3
  n = len(names)
  rows_n = (n + cols - 1) // cols
  fig, axes = _subplots(cols, rows_n, w=3.6, h=2.6, sharex=True, sharey=True)
  for i, name in enumerate(names):
    ax = axes[i // cols][i % cols]
    w = np.stack([data_loader.window_recording(r.samples)
                  for r in recs if r.label == name])
    t = np.arange(w.shape[1]) / 100.0
    for a, c in zip(range(3), _PALETTE[:3]):
      m, sd = w[:, :, a].mean(0), w[:, :, a].std(0)
      ax.plot(t, m, color=c, lw=1.3, label=f"a{'xyz'[a]}")
      ax.fill_between(t, m - sd, m + sd, color=c, alpha=0.18, lw=0)
    ax.set_title(f"{name} (n={len(w)})")
    ax.set_ylim(-2.5, 2.5)
    if i // cols == rows_n - 1:
      ax.set_xlabel("time (s)")
    if i % cols == 0:
      ax.set_ylabel("acceleration (g)")
  for j in range(n, rows_n * cols):
    axes[j // cols][j % cols].set_visible(False)
  axes[0][0].legend(loc="upper right", ncol=3)
  _finish(fig, os.path.join(figs, "class_signatures.png"))

  return "\n".join([
      "## Dataset & Physical Separability", "",
      f"Fingerprint `{splits.fingerprint}` | **{len(recs)}** recordings"
      f" ({len(splits.x_train)} train / {len(splits.x_val)} val /"
      f" {len(splits.x_test)} frozen test) | `{data_loader.WINDOW_ROWS}x3`"
      f" window @ 100 Hz | raw length {lengths.min()}–{lengths.max()} rows"
      f" (median {int(np.median(lengths))}).",
      "- **Pose-separable gestures**: `Roll` flips `Z` gravity negative,"
      " `Twirl` and `Punch` hold the cable vertical (`+X ~ +0.8g, Z ~ 0g`),"
      " and `Poke` maintains a tilted firm-grip baseline (`+Y ~ +0.75g`).",
      "- **Impulse & rhythm-separable gestures**: `Slap` reaches `~15g` peak"
      " dynamic impact with high jerk, `Flick` snaps and rebounds along the"
      " cable, `Shake` oscillates at `~4-8 Hz`, `Pendulum` swings smoothly,"
      " and `Null` stays below `~0.5g` dynamic amplitude.", "",
      _md_table([
          "class", "train", "val", "test", "total",
          "mean pose (X,Y,Z) g", "peak |a_dyn| g", "RMS jerk g/s", "dom Hz",
      ], rows), "",
      _caption("dataset_hist", "Recording length and acceleration magnitude"
               " distributions (1 g mode at rest)."), "",
      _caption("kinematic_space", "(a) Gravity pose vs. peak dynamic impulse"
               " across all 120 recordings; (b) per-class physical feature"
               " matrix explaining why raw pose + centered + jerk channels"
               " separate the 10 classes."), "",
      _caption("class_signatures", "Per-class mean ± 1 SD acceleration over the"
               " 1 s window."), "",
  ])


def layer_table(model_bytes: bytes) -> tuple[list[list[Any]], dict[str, Any]]:
  import tensorflow as tf  # pylint: disable=import-outside-toplevel
  interp = tf.lite.Interpreter(
      model_content=model_bytes, num_threads=1,
      experimental_op_resolver_type=(
          tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
      ))
  interp.allocate_tensors()
  tensors = {d["index"]: d for d in interp.get_tensor_details()}
  ops = interp._get_ops_details()  # pylint: disable=protected-access
  acts = {d["index"] for d in interp.get_input_details()}
  for op in ops:
    acts.update(int(i) for i in op["outputs"])
  rows = []
  tot_macs = tot_w = 0
  for k, op in enumerate(ops):
    name = op["op_name"]
    ins = [tensors[int(i)] for i in op["inputs"] if int(i) in tensors]
    outs = [tensors[int(i)] for i in op["outputs"] if int(i) in tensors]
    out_elems = sum(int(np.prod(d["shape"])) for d in outs)
    macs = out_elems
    if name == "FULLY_CONNECTED" and len(ins) >= 2:
      macs = out_elems * int(ins[1]["shape"][-1])
    elif name == "CONV_2D" and len(ins) >= 2:
      macs = out_elems * int(np.prod(ins[1]["shape"][1:]))
    elif name == "DEPTHWISE_CONV_2D" and len(ins) >= 2:
      macs = out_elems * int(ins[1]["shape"][1]) * int(ins[1]["shape"][2])
    wbytes = sum(int(np.prod(d["shape"])) * np.dtype(d["dtype"]).itemsize
                 for d in ins[1:] if d["index"] not in acts)
    live_bytes = sum(int(np.prod(d["shape"])) * np.dtype(d["dtype"]).itemsize
                     for d in ins + outs if d["index"] in acts)
    tot_macs += macs
    tot_w += wbytes
    shape = "x".join(str(int(v)) for v in outs[0]["shape"]) if outs else ""
    dtype = np.dtype(outs[0]["dtype"]).name if outs else ""
    rows.append([k, name, shape, dtype, macs, wbytes, live_bytes])
  inp = interp.get_input_details()[0]
  out = interp.get_output_details()[0]
  meta = {
      "total_macs": tot_macs, "total_weight_bytes": tot_w,
      "input": f"{list(map(int, inp['shape']))} {np.dtype(inp['dtype']).name}"
               f" scale={inp['quantization'][0]:.5g} zp={inp['quantization'][1]}",
      "output": f"{list(map(int, out['shape']))} {np.dtype(out['dtype']).name}"
                f" scale={out['quantization'][0]:.5g}"
                f" zp={out['quantization'][1]}",
  }
  return rows, meta


def model_section(model_bytes: bytes, artifact: Mapping[str, Any],
                  figs: str, splits: data_loader.Splits | None = None) -> str:
  rows, meta = layer_table(model_bytes)
  budget_cols = [
      "metric", "model KB", "arena KB (est)", "MFLOPs", "params (k)",
      "MCU ms (est)", "host ms",
  ]
  budget_rows = [
      ["value", artifact["model_kb"], artifact["arena_kb"], artifact["mflops"],
       artifact["params_k"], artifact["mcu_ms_est"],
       artifact["host_latency_ms"]],
      ["soft / hard budget",
       f"{deploy.SOFT_MODEL_KB} / {deploy.MAX_MODEL_KB}",
       f"{deploy.SOFT_ARENA_KB} / {deploy.MAX_ARENA_KB}",
       f"{deploy.SOFT_MFLOPS} / -", f"{deploy.SOFT_PARAMS_K} / -",
       f"{deploy.SOFT_MCU_MS} / -", f"{deploy.SOFT_HOST_LATENCY_MS} / -"],
  ]
  fig, axes = _subplots(3, 1, w=3.8, h=max(3.0, 0.26 * len(rows) + 1.1))
  labels = [f"{r[0]:02d} {r[1]}" for r in rows]
  macs = np.array([r[4] for r in rows], float)
  wbytes = np.array([r[5] for r in rows], float)
  act_kb = np.array([r[6] for r in rows], float) / 1024.0
  y = np.arange(len(rows))
  ax = axes[0][0]
  ax.barh(y, macs / 1e3, color=_PALETTE[0])
  ax.set_yticks(y)
  ax.set_yticklabels(labels, fontsize=7.5)
  ax.invert_yaxis()
  ax.set_xlabel("kMACs per inference")
  ax.set_title("Compute per op")
  ax = axes[0][1]
  ax.barh(y, wbytes / 1024.0, color=_PALETTE[1])
  ax.set_yticks(y)
  ax.set_yticklabels([""] * len(rows))
  ax.invert_yaxis()
  ax.set_xlabel("weight size (KB)")
  ax.set_title("Weights per op")
  ax = axes[0][2]
  ax.barh(y, act_kb, color=_PALETTE[2])
  ax.set_yticks(y)
  ax.set_yticklabels([""] * len(rows))
  ax.invert_yaxis()
  ax.set_xlabel("live activations (KB)")
  ax.set_title("Peak RAM pressure per op")
  _finish(fig, os.path.join(figs, "macs_per_op.png"))

  macs_cap = _caption(
      "macs_per_op",
      "Per-op (a) compute (kMACs), (b) weight footprint (KB), and (c) live"
      " activation RAM pressure (KB) in execution order.",
  )

  internals_cap = []
  if splits is not None:
    try:
      import tensorflow as tf  # pylint: disable=import-outside-toplevel
      from sklearn import manifold  # pylint: disable=import-outside-toplevel
      plt = _plt()
      cmap = plt.get_cmap("tab10")
      names = splits.class_names
      x_all = pd.concat([splits.x_learning, splits.x_test], ignore_index=True)
      y_all = pd.concat([splits.y_learning, splits.y_test],
                        ignore_index=True).iloc[:, 0].to_numpy(int)
      is_test = np.zeros(len(x_all), dtype=bool)
      is_test[len(splits.x_learning):] = True
      feats_all = data_loader.features_only(x_all)

      interp_p = tf.lite.Interpreter(
          model_content=model_bytes, num_threads=1,
          experimental_preserve_all_tensors=True,
      )
      interp_p.allocate_tensors()
      in_det = interp_p.get_input_details()[0]
      out_det = interp_p.get_output_details()[0]
      t_map = {d["index"]: d for d in interp_p.get_tensor_details()}
      ops_p = interp_p._get_ops_details()  # pylint: disable=protected-access
      fc_ops = [op for op in ops_p if op["op_name"] == "FULLY_CONNECTED"]
      pen_idx = int(fc_ops[-1]["inputs"][0]) if fc_ops else int(out_det["index"])
      logit_idx = int(fc_ops[-1]["outputs"][0]) if fc_ops else int(out_det["index"])

      def _deq(idx: int) -> np.ndarray:
        raw = interp_p.get_tensor(idx).astype(np.float32)
        sc, zp = t_map[idx].get("quantization", (0.0, 0))
        return (raw - float(zp)) * float(sc) if sc else raw

      def _eval_batch(batch_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        in_shape = tuple(int(v) for v in in_det["shape"])
        in_dtype = in_det["dtype"]
        in_sc, in_zp = in_det["quantization"]
        pen_list, log_list = [], []
        for row_f in batch_2d:
          arr = row_f.astype(np.float32).reshape(in_shape)
          if in_dtype in (np.int8, np.uint8) and in_sc > 0:
            info = np.iinfo(in_dtype)
            arr = np.clip(np.round(arr / in_sc + in_zp),
                          info.min, info.max).astype(in_dtype)
          else:
            arr = arr.astype(in_dtype)
          interp_p.set_tensor(in_det["index"], arr)
          interp_p.invoke()
          pen_list.append(_deq(pen_idx).reshape(-1))
          log_list.append(_deq(logit_idx).reshape(-1))
        return np.stack(pen_list), np.stack(log_list)

      pen_all, _ = _eval_batch(feats_all)
      emb_2d = manifold.TSNE(
          n_components=2, perplexity=12, init="pca",
          learning_rate="auto", random_state=42,
      ).fit_transform(pen_all)

      fig_int, axes_int = _subplots(2, 1, w=5.2, h=4.2)
      ax_pca = axes_int[0][0]
      for i, n in enumerate(names):
        m_tr = (y_all == i) & (~is_test)
        m_te = (y_all == i) & is_test
        if m_tr.any():
          ax_pca.scatter(emb_2d[m_tr, 0], emb_2d[m_tr, 1], color=cmap(i % 10),
                         s=24, alpha=0.70, label=n)
        if m_te.any():
          ax_pca.scatter(emb_2d[m_te, 0], emb_2d[m_te, 1], color=cmap(i % 10),
                         marker="*", s=100, edgecolors="k", lw=0.55, zorder=4)
      ax_pca.set_xlabel("t-SNE dimension 1 (penultimate int8 features)")
      ax_pca.set_ylabel("t-SNE dimension 2")
      ax_pca.set_title(
          f"Penultimate feature space ({pen_all.shape[1]}D → 2D; ★ = test)"
      )
      ax_pca.legend(loc="best", ncol=2, fontsize=7.5)

      # Panel (b): Temporal occlusion sensitivity on pre-softmax logit margin
      ax_sal = axes_int[0][1]
      x_te_arr = data_loader.features_only(splits.x_test).reshape(-1, 100, 3)
      y_te_arr = splits.y_test.iloc[:, 0].to_numpy(int)
      win_sz, step_sz = 20, 5
      centers_s = (
          np.arange(0, 100 - win_sz + 1, step_sz) + win_sz / 2.0
      ) / 100.0
      highlight = [c for c in ("Drop", "Slap", "Roll", "Shake", "Twirl")
                   if c in names]
      for c_name in highlight:
        c_idx = names.index(c_name)
        idxs = np.where(y_te_arr == c_idx)[0]
        if not len(idxs):
          continue
        # Average occlusion curve across test examples of this class
        curves = []
        for idx_i in idxs:
          w0 = x_te_arr[idx_i].copy()
          base_pose = w0.mean(axis=0, keepdims=True)
          occluded = [w0.reshape(-1)]
          for st_i in range(0, 100 - win_sz + 1, step_sz):
            w_occ = w0.copy()
            w_occ[st_i:st_i + win_sz] = base_pose
            occluded.append(w_occ.reshape(-1))
          _, occ_logits = _eval_batch(np.stack(occluded))
          other_mask = np.arange(occ_logits.shape[1]) != c_idx
          margins = occ_logits[:, c_idx] - occ_logits[:, other_mask].max(axis=1)
          base_m = max(1e-3, abs(float(margins[0])))
          drop_m = np.clip((margins[0] - margins[1:]) / base_m, 0.0, 1.5)
          curves.append(drop_m)
        mean_curve = np.mean(curves, axis=0)
        ax_sal.plot(centers_s, mean_curve, lw=1.7, color=cmap(c_idx % 10),
                    marker="o", markersize=3.5, label=c_name)
      ax_sal.set_xlabel("occluded 200 ms window center (s)")
      ax_sal.set_ylabel("relative logit-margin drop")
      ax_sal.set_title("Temporal occlusion sensitivity (where the model looks)")
      ax_sal.legend(loc="upper right", fontsize=8)
      _finish(fig_int, os.path.join(figs, "model_internals.png"))
      internals_cap = [
          _caption("model_internals",
                   f"(a) t-SNE projection of the quantized {pen_all.shape[1]}D"
                   " penultimate feature representation across all 120"
                   " recordings (stars = frozen test); (b) sliding 200 ms"
                   " temporal occlusion sensitivity on the true-class logit"
                   " margin."), "",
      ]
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("model_internals plot failed: %s", e)

  return "\n".join([
      "## Model Architecture, Internals & Budgets", "",
      f"- **I/O & Quantization**: input `{meta['input']}` → output"
      f" `{meta['output']}` | int8={artifact['quantized']}"
      f" ({artifact['int8_fraction']:.0%} int8 ops) | **{meta['total_macs']:,}**"
      f" MACs, **{meta['total_weight_bytes']:,}** weight bytes",
      f"- **Operators ({len(artifact['ops'])})**:"
      f" `{', '.join(artifact['ops'])}`", "",
      _md_table(budget_cols, budget_rows), "",
      "### Layer Breakdown", "",
      _md_table(
          ["#", "op", "output", "dtype", "MACs", "weight B", "live act B"],
          rows,
      ),
      "",
      macs_cap, "",
      *internals_cap,
  ])


def host_benchmark_section(model_bytes: bytes, features: np.ndarray,
                           figs: str) -> str:
  import platform  # pylint: disable=import-outside-toplevel
  import tensorflow as tf  # pylint: disable=import-outside-toplevel
  res = {}
  for label, kwargs in (
      ("TFLite reference (no delegate)", dict(
          experimental_op_resolver_type=tf.lite.experimental.OpResolverType
          .BUILTIN_WITHOUT_DEFAULT_DELEGATES)),
      ("TFLite + XNNPACK", {}),
  ):
    interp = tf.lite.Interpreter(model_content=model_bytes, num_threads=1,
                                 **kwargs)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    x = np.zeros(inp["shape"], dtype=inp["dtype"])
    interp.set_tensor(inp["index"], x)
    for _ in range(20):
      interp.invoke()
    lat = []
    for _ in range(300):
      t0 = time.perf_counter()
      interp.invoke()
      lat.append((time.perf_counter() - t0) * 1e3)
    res[label] = np.array(lat)
  t0 = time.perf_counter()
  deploy.run_tflite(model_bytes, features)
  batch_ms = (time.perf_counter() - t0) * 1e3 / max(1, len(features))

  fig, axes = _subplots(1, 1, w=6.4, h=2.8)
  ax = axes[0][0]
  names_ = list(res.keys())
  med = [np.median(v) for v in res.values()]
  err = [[m - np.percentile(v, 5) for m, v in zip(med, res.values())],
         [np.percentile(v, 95) - m for m, v in zip(med, res.values())]]
  ax.barh(names_, med, xerr=err, color=_PALETTE[:len(names_)], capsize=3)
  ax.set_xlabel("latency per inference (ms), median with 5th to 95th pct")
  ax.set_title(f"Host latency, {platform.machine()}, single thread")
  _finish(fig, os.path.join(figs, "host_latency.png"))

  rows = [[k, np.median(v), np.percentile(v, 95), v.min(), 1000.0 / np.median(v)]
          for k, v in res.items()]
  return "\n".join([
      "## Host Benchmark", "",
      f"`Python {platform.python_version()}` | `TF {tf.__version__}` |"
      f" `{platform.platform()}` | Evaluator e2e: **{batch_ms:.3f} ms/row**.",
      "",
      _md_table(["runtime", "p50 ms", "p95 ms", "min ms", "inf/s"], rows), "",
      _caption("host_latency", "Single-threaded host latency over 300"
               " invocations (reference kernel vs XNNPACK)."), "",
  ])


def evaluation_section(model_bytes: bytes, splits: data_loader.Splits,
                       figs: str) -> tuple[str, dict[str, Any]]:
  from sklearn import metrics as skm  # pylint: disable=import-outside-toplevel
  names = splits.class_names
  out: dict[str, Any] = {}
  reps: dict[str, Any] = {}
  fig, axes2 = _subplots(2, 1, w=5.0, h=4.6)
  split_rows = []
  for ax, (tag, x, y) in zip(
      axes2[0],
      (("val", splits.x_val_eval, splits.y_val_eval),
       ("test", splits.x_test_eval, splits.y_test_eval)),
  ):
    feats = data_loader.features_only(x)
    scores = deploy.run_tflite(model_bytes, feats)
    pred = scores.argmax(1)
    truth = y.iloc[:, 0].to_numpy(int)
    cm = skm.confusion_matrix(truth, pred, labels=range(len(names)))
    rep = skm.classification_report(truth, pred, labels=range(len(names)),
                                    target_names=names, output_dict=True,
                                    zero_division=0)
    reps[tag] = rep
    conf = scores.max(1)
    c_ok = float(conf[pred == truth].mean()) if (pred == truth).any() else float("nan")
    c_err = float(conf[pred != truth].mean()) if (pred != truth).any() else float("nan")
    out[tag] = {
        "f1_macro": float(skm.f1_score(truth, pred, average="macro",
                                       zero_division=0)),
        "accuracy": float((pred == truth).mean()),
        "confusion": cm.tolist(),
        "max_prob_mean": float(conf.mean()),
        "confidence_correct": c_ok,
        "confidence_wrong": c_err,
    }
    split_rows.append([
        tag, len(truth), out[tag]["accuracy"], out[tag]["f1_macro"],
        f"{out[tag]['max_prob_mean']:.3f} (ok {c_ok:.3f}, err {c_err:.3f})",
    ])
    row_sum = cm.sum(1, keepdims=True)
    norm = np.divide(cm, np.maximum(row_sum, 1))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
      for j in range(len(names)):
        if cm[i, j]:
          ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                  color="white" if norm[i, j] > 0.5 else "black")
    ax.set_title(f"{tag}: accuracy {out[tag]['accuracy']:.2f},"
                 f" macro F1 {out[tag]['f1_macro']:.2f}")
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.grid(False)
  for ax in axes2[0]:
    ax.grid(False)
  fig.tight_layout()
  for i, ax in enumerate(axes2[0]):
    ax.text(-0.08, 1.06, f"({chr(97 + i)})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="right")
  fig.savefig(os.path.join(figs, "confusion.png"), dpi=DPI, bbox_inches="tight")
  _plt().close(fig)
  class_rows = [
      [
          n,
          f"{reps['val'][n]['f1-score']:.3g} (n={int(reps['val'][n]['support'])})",
          reps["test"][n]["precision"],
          reps["test"][n]["recall"],
          f"{reps['test'][n]['f1-score']:.3g} (n={int(reps['test'][n]['support'])})",
      ]
      for n in names
  ]
  text = "\n".join([
      "## TFLite Evaluation (Validation and Frozen Test)", "",
      _md_table(["split", "rows", "accuracy", "macro F1",
                 "mean top-1 prob (all / ok / err)"], split_rows), "",
      _md_table(["class", "val F1 (support)", "test precision", "test recall",
                 "test F1 (support)"], class_rows), "",
      _caption("confusion", "Row-normalized confusion matrices on validation (a)"
               " and frozen test (b) splits; cell labels are counts."), "",
  ])
  return text, out


def curves_section(model_bytes: bytes, splits: data_loader.Splits,
                   figs: str) -> tuple[str, dict[str, Any]]:
  """ROC, PR, calibration, and per-class bars on all held-out rows."""
  from sklearn import metrics as skm  # pylint: disable=import-outside-toplevel
  from sklearn import preprocessing  # pylint: disable=import-outside-toplevel
  names = splits.class_names
  k = len(names)
  x = pd.concat([splits.x_val_eval, splits.x_test_eval]).reset_index(drop=True)
  y = pd.concat([splits.y_val_eval, splits.y_test_eval]).reset_index(drop=True)
  truth = y.iloc[:, 0].to_numpy(int)
  scores = deploy.run_tflite(model_bytes, data_loader.features_only(x))
  if scores.shape[1] < k:  # artifact from an older, smaller class set
    scores = np.hstack([scores, np.zeros((len(scores), k - scores.shape[1]))])
  # int8 softmax outputs may not sum exactly to 1; renormalize for curves.
  prob = np.clip(scores, 0, None)
  prob = prob / np.clip(prob.sum(1, keepdims=True), 1e-9, None)
  pred = prob.argmax(1)
  onehot = preprocessing.label_binarize(truth, classes=range(k))
  if k == 2:
    onehot = np.hstack([1 - onehot, onehot])

  plt = _plt()
  fig, axes = _subplots(2, 2, w=5.0, h=4.3)
  rows = []
  cmap = plt.get_cmap("tab10")
  for i, n in enumerate(names):
    present = onehot[:, i].any() and (onehot[:, i] == 0).any()
    auc = ap = float("nan")
    if present:
      fpr, tpr, _ = skm.roc_curve(onehot[:, i], prob[:, i])
      auc = skm.auc(fpr, tpr)
      pr, rc, _ = skm.precision_recall_curve(onehot[:, i], prob[:, i])
      ap = skm.average_precision_score(onehot[:, i], prob[:, i])
      axes[0][0].plot(fpr, tpr, color=cmap(i % 10), lw=1.3,
                      label=f"{n} (AUC {auc:.2f})")
      axes[0][1].plot(rc, pr, color=cmap(i % 10), lw=1.3,
                      label=f"{n} (AP {ap:.2f})")
    sup = int(onehot[:, i].sum())
    p_ = skm.precision_score(truth == i, pred == i, zero_division=0)
    r_ = skm.recall_score(truth == i, pred == i, zero_division=0)
    f_ = skm.f1_score(truth == i, pred == i, zero_division=0)
    rows.append([n, sup, p_, r_, f_, auc, ap])
  axes[0][0].plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
  axes[0][0].set_xlabel("false positive rate")
  axes[0][0].set_ylabel("true positive rate")
  axes[0][0].set_title("One-vs-rest ROC")
  axes[0][0].legend(loc="lower right")
  axes[0][1].set_xlabel("recall")
  axes[0][1].set_ylabel("precision")
  axes[0][1].set_title("One-vs-rest precision-recall")
  axes[0][1].legend(loc="lower left")

  # Reliability diagram and expected calibration error on the top-1 prob.
  conf = prob.max(1)
  correct = (pred == truth).astype(float)
  bins = np.linspace(0, 1, 11)
  idx = np.clip(np.digitize(conf, bins) - 1, 0, 9)
  acc_b, conf_b, cnt_b = np.zeros(10), np.zeros(10), np.zeros(10)
  for b in range(10):
    m = idx == b
    cnt_b[b] = m.sum()
    if m.any():
      acc_b[b], conf_b[b] = correct[m].mean(), conf[m].mean()
  ece = float(np.sum(cnt_b / max(1, len(conf)) * np.abs(acc_b - conf_b)))
  centers = (bins[:-1] + bins[1:]) / 2
  axes[1][0].bar(centers, acc_b, width=0.1, color=_PALETTE[0], alpha=0.85,
                 edgecolor="white", label="observed accuracy")
  axes[1][0].plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
  axes[1][0].legend(loc="upper left")
  for c, n_, a in zip(centers, cnt_b, acc_b):
    if n_:
      axes[1][0].text(c, a + 0.02, int(n_), ha="center", fontsize=7)
  axes[1][0].set_xlabel("predicted confidence (top-1)")
  axes[1][0].set_ylabel("observed accuracy")
  axes[1][0].set_title(f"Reliability, ECE = {ece:.3f}")
  axes[1][0].set_xlim(0, 1)
  axes[1][0].set_ylim(0, 1.05)

  axes[1][1].hist([conf[correct == 1], conf[correct == 0]], bins=20,
                  range=(0, 1), stacked=True,
                  color=[_PALETTE[2], _PALETTE[1]], edgecolor="white",
                  label=["correct", "wrong"])
  axes[1][1].set_xlabel("top-1 confidence")
  axes[1][1].set_ylabel("windows")
  axes[1][1].set_title("Confidence of predictions")
  axes[1][1].legend(loc="upper left")
  _finish(fig, os.path.join(figs, "curves.png"))

  fig, axes_pc = _subplots(1, 1, w=max(6.0, 0.9 * k + 2), h=3.2)
  ax = axes_pc[0][0]
  xi = np.arange(k)
  for j, (lab, col) in enumerate((("precision", _PALETTE[0]),
                                  ("recall", _PALETTE[2]), ("F1", _PALETTE[1]))):
    ax.bar(xi + (j - 1) * 0.27, [r[2 + j] for r in rows], width=0.27,
           color=col, label=lab, edgecolor="white")
  ax.set_xticks(xi)
  ax.set_xticklabels(names, rotation=30, ha="right")
  ax.set_ylim(0, 1.05)
  ax.set_ylabel("score")
  ax.legend(ncol=3, loc="lower right")
  ax.set_title("Per-class precision, recall, and F1")
  _finish(fig, os.path.join(figs, "per_class.png"))

  macro_auc = float(np.nanmean([r[5] for r in rows]))
  macro_ap = float(np.nanmean([r[6] for r in rows]))
  try:
    ll = float(skm.log_loss(truth, prob, labels=range(k)))
  except ValueError:
    ll = float("nan")
  mcc = float(skm.matthews_corrcoef(truth, pred))
  kappa = float(skm.cohen_kappa_score(truth, pred))
  top2 = float(np.mean([t in np.argsort(-p)[:2] for t, p in zip(truth, prob)]))
  out = {"rows": len(truth), "macro_auc": macro_auc, "macro_ap": macro_ap,
         "ece": ece, "log_loss": ll, "mcc": mcc, "kappa": kappa,
         "top2_accuracy": top2,
         "accuracy": float(correct.mean()),
         "f1_macro": float(skm.f1_score(truth, pred, average="macro",
                                        zero_division=0))}
  agg_headers = [
      "accuracy", "macro F1", "top-2 acc", "macro AUC", "macro AP",
      "MCC", "kappa", "log loss", "ECE (10 bins)",
  ]
  agg_row = [[
      out["accuracy"], out["f1_macro"], top2, macro_auc, macro_ap,
      mcc, kappa, ll, ece,
  ]]
  text = "\n".join([
      f"## Classifier Calibration & ROC/PR (Held-Out Val + Test, {len(truth)} rows)",
      "",
      _md_table(agg_headers, agg_row), "",
      _md_table(["class", "support", "precision", "recall", "f1", "ROC AUC",
                 "AP"], rows), "",
      _caption("curves", "(a) One-vs-rest ROC, (b) precision-recall,"
               " (c) calibration reliability diagram, (d) top-1 confidence."),
      "",
      _caption("per_class", "Per-class precision, recall, and F1 on all"
               " held-out rows."), "",
  ])
  return text, out


def stream_section(model_bytes: bytes, splits: data_loader.Splits,
                   figs: str) -> tuple[str, dict[str, Any]]:
  """Sliding-window simulation over concatenated test recordings."""
  names = splits.class_names
  recs = data_loader.list_recordings(
      splits.data_dir or data_loader.default_data_dir())
  test_ids = set(splits.x_test[data_loader.GROUP_COLUMN])
  rng = np.random.default_rng(0)
  test_recs = [r for r in recs if r.recording_id in test_ids]
  rng.shuffle(test_recs)
  w = data_loader.WINDOW_ROWS
  # Normalize each recording to exactly one window so windows at offsets
  # k*w are pure and those at k*w + w/2 straddle a boundary.
  stream = np.concatenate([data_loader.window_recording(r.samples, w)
                           for r in test_recs])
  truth = np.concatenate([[names.index(r.label)] * w for r in test_recs])
  step = w // 2
  starts = list(range(0, len(stream) - w + 1, step))
  windows = np.stack([stream[s:s + w].reshape(-1) for s in starts])
  scores = deploy.run_tflite(model_bytes, windows)
  pred = scores.argmax(1)
  conf = scores.max(1)
  # majority truth label of each window
  wt = np.array([np.bincount(truth[s:s + w], minlength=len(names)).argmax()
                 for s in starts])
  pure = np.array([len(np.unique(truth[s:s + w])) == 1 for s in starts])
  acc_pure = float((pred[pure] == wt[pure]).mean()) if pure.any() else float("nan")
  null_idx = names.index("Null") if "Null" in names else None
  fp_null = float("nan")
  if null_idx is not None and (wt[pure] == null_idx).any():
    m = pure & (wt == null_idx)
    fp_null = float((pred[m] != null_idx).mean())
  # transitions: how many windows after a boundary until prediction is right
  flips = int(np.sum(pred[1:] != pred[:-1]))

  fig, axes = _subplots(1, 2, w=10.0, h=2.3, sharex=True,
                        gridspec_kw={"height_ratios": [2, 1]})
  t = np.array(starts) / 100.0
  ax = axes[0][0]
  ax.step(np.arange(len(truth)) / 100.0, truth, where="post", color="k",
          lw=1, label="true class")
  ok = pred == wt
  ax.scatter(t[ok] + w / 200.0, pred[ok], color=_PALETTE[2], s=18, zorder=3,
             label="correct window")
  ax.scatter(t[~ok] + w / 200.0, pred[~ok], color=_PALETTE[1], s=18, zorder=3,
             marker="x", label="wrong window")
  ax.set_yticks(range(len(names)))
  ax.set_yticklabels(names, fontsize=8)
  ax.legend(loc="upper right", ncol=3)
  ax.set_title("Sliding-window predictions over concatenated test recordings")
  ax = axes[1][0]
  ax.plot(t + w / 200.0, conf, color=_PALETTE[0], lw=1.2)
  ax.axhline(0.6, color="k", ls="--", lw=0.8, label="UNSURE threshold")
  ax.set_ylim(0, 1.02)
  ax.set_ylabel("top-1 prob")
  ax.set_xlabel("time (s)")
  ax.legend(loc="lower right")
  _finish(fig, os.path.join(figs, "stream.png"))
  out = {"windows": len(starts), "pure_window_accuracy": acc_pure,
         "null_false_trigger_rate": fp_null, "prediction_flips": flips,
         "mixed_windows": int((~pure).sum())}
  stream_headers = [
      "windows", "duration (s)", "pure-window acc", "Null FPR",
      "boundary windows", "prediction flips",
  ]
  stream_row = [[
      len(starts), f"{len(stream) / 100:.1f}", acc_pure, fp_null,
      out["mixed_windows"], flips,
  ]]
  text = "\n".join([
      "## Streaming Simulation (50% Overlap)", "",
      _md_table(stream_headers, stream_row), "",
      _caption("stream", "Sliding 1 s windows (50% overlap) over concatenated"
               " test recordings: (a) predicted vs true class, (b) top-1"
               " confidence vs UNSURE threshold."), "",
  ])
  return text, out


def _pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
  """Non-dominated set for (cost, quality): lower cost, higher quality."""
  pts = sorted(set(points))
  front, best = [], -1.0
  for c, q in pts:
    if q > best:
      front.append((c, q))
      best = q
  return front


def pareto_section(run_dir: str, summary: Mapping[str, Any],
                   figs: str) -> str:
  """Accuracy, compute, memory, and score tradeoffs over all scored attempts."""
  base = os.path.dirname(os.path.dirname(run_dir))
  this_fp = summary["dataset_fingerprint"]

  def _run_meta(rd: str) -> tuple[str, bool]:
    fp, cv = "", False
    log = os.path.join(rd, "run.log")
    if os.path.isfile(log):
      with open(log, encoding="utf-8", errors="ignore") as lf:
        for line in lf:
          m = re.search(r"dataset (\w+):", line)
          if m and not fp:
            fp = m.group(1)
          if "Cycle scoring:" in line:
            cv = True
          if fp and cv:
            break
    return fp, cv

  this_cv = _run_meta(run_dir)[1]
  rows = []
  for f in glob.glob(os.path.join(base, "*", "*", "students", "*.csv")):
    rd = os.path.dirname(os.path.dirname(f))
    if os.path.isfile(os.path.join(rd, "MOCK")):
      continue
    fp, cv = _run_meta(rd)
    if fp != this_fp or cv != this_cv:
      continue  # only comparable attempts: same data, same scoring protocol
    run = os.path.basename(rd)
    exp = os.path.basename(os.path.dirname(rd))
    try:
      df = pd.read_csv(f)
    except Exception:  # pylint: disable=broad-exception-caught
      continue
    need = {"metrics_f1_macro", "metrics_model_kb", "metrics_mflops"}
    if not need <= set(df.columns):
      continue
    df = df[pd.to_numeric(df["overall_score"], errors="coerce").notna()]
    for _, r in df.iterrows():
      sc_v = float(r["overall_score"])
      if not np.isfinite(sc_v):
        continue
      rows.append({
          "experiment": exp, "run": run,
          "student": os.path.basename(f)[:-4].replace("student_", ""),
          "cycle": int(r["cycle"]), "score": sc_v,
          "f1": float(r["metrics_f1_macro"]), "kb": float(r["metrics_model_kb"]),
          "mflops": float(r["metrics_mflops"]),
          "arena": float(r.get("metrics_arena_kb", float("nan"))),
          "this_run": os.path.basename(run_dir) == run,
      })
  if not rows:
    return ""
  df = pd.DataFrame(rows)
  art = summary["artifact"]
  dep = (art["model_kb"], summary["test_metrics"].get("f1_macro", float("nan")))

  fig, axes = _subplots(2, 2, w=5.2, h=4.1)
  for ax, cost, label, budget in (
      (axes[0][0], "kb", "model size (KB)", deploy.SOFT_MODEL_KB),
      (axes[0][1], "mflops", "compute (MFLOPs per inference)", deploy.SOFT_MFLOPS),
  ):
    other = df[~df.this_run]
    mine = df[df.this_run]
    ax.scatter(other[cost], other.f1, s=16, color="#B0B0B0", alpha=0.8,
               label="earlier runs")
    ax.scatter(mine[cost], mine.f1, s=28, color=_PALETTE[0], alpha=0.9,
               label="this run")
    front = _pareto_front(list(zip(df[cost], df.f1)))
    ax.step([p[0] for p in front], [p[1] for p in front], where="post",
            color=_PALETTE[1], lw=1.6, label="Pareto frontier")
    ax.scatter([p[0] for p in front], [p[1] for p in front], s=30,
               color=_PALETTE[1], zorder=4)
    ax.axvline(budget, color="k", ls=":", lw=0.9, label=f"soft budget ({budget:g})")
    if cost == "kb":
      ax.scatter([dep[0]], [dep[1]], marker="*", s=220, color=_PALETTE[4],
                 edgecolor="k", zorder=5, label="exported model (test F1)")
    ax.set_xlabel(label)
    ax.set_ylabel("macro F1 (out-of-fold validation)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=7.5)
  axes[0][0].set_title("Accuracy vs. Flash Size")
  axes[0][1].set_title("Accuracy vs. Compute (MACs)")

  # Panel (c): Activation Arena KB vs Model KB colored by Validation F1
  ax_mem = axes[1][0]
  df_mem = df[df.arena.notna()]
  if not df_mem.empty:
    sc_plot = ax_mem.scatter(df_mem.kb, df_mem.arena, c=df_mem.f1,
                             cmap="viridis", vmin=0.65, vmax=1.0, s=26,
                             alpha=0.9, edgecolors="k", lw=0.3)
    cbar = fig.colorbar(sc_plot, ax=ax_mem, fraction=0.046, pad=0.04)
    cbar.ax.set_label("cbar")
    cbar.set_label("val macro F1", fontsize=8)
  ax_mem.scatter([art["model_kb"]], [art["arena_kb"]], marker="*", s=220,
                 color=_PALETTE[4], edgecolor="k", zorder=5,
                 label="exported model")
  ax_mem.axvline(deploy.SOFT_MODEL_KB, color="k", ls=":", lw=0.8)
  ax_mem.axhline(deploy.SOFT_ARENA_KB, color="k", ls="--", lw=0.8,
                 label=f"arena budget ({deploy.SOFT_ARENA_KB:g} KB)")
  ax_mem.set_xlabel("model flash size (KB)")
  ax_mem.set_ylabel("estimated activation arena (KB)")
  ax_mem.set_title("Hardware Memory Envelope (RAM vs. Flash)")
  ax_mem.legend(loc="upper left", fontsize=7.5)

  # Panel (d): Overall Score vs Model Size showing the 8-14 KB sweet spot
  ax_sc = axes[1][1]
  other = df[~df.this_run]
  mine = df[df.this_run]
  ax_sc.scatter(other.kb, other.score, s=16, color="#B0B0B0", alpha=0.8,
                label="earlier runs")
  ax_sc.scatter(mine.kb, mine.score, s=28, color=_PALETTE[2], alpha=0.9,
                label="this run")
  front_sc = _pareto_front(list(zip(df.kb, df.score)))
  ax_sc.step([p[0] for p in front_sc], [p[1] for p in front_sc], where="post",
             color=_PALETTE[1], lw=1.6, label="Score frontier")
  ax_sc.axvspan(8.0, 14.5, color=_PALETTE[2], alpha=0.10,
                label="8–14.5 KB sweet spot")
  ax_sc.set_xlabel("model size (KB)")
  ax_sc.set_ylabel("overall validation score")
  ax_sc.set_ylim(max(0.5, float(df.score.quantile(0.05)) - 0.03), 0.96)
  ax_sc.set_title("Overall Score vs. Flash Size Trade-off")
  ax_sc.legend(loc="lower right", fontsize=7.5)
  _finish(fig, os.path.join(figs, "pareto.png"))

  front_kb = _pareto_front(list(zip(df.kb, df.f1)))
  table = []
  for kb, f1 in front_kb:
    r = df[(df.kb == kb) & (df.f1 == f1)].iloc[0]
    table.append([f"{kb:.1f}", f"{f1:.3f}", f"{r.mflops:.3f}",
                  f"{r.arena:.1f}", f"{r.score:.4f}", r.run, r.student,
                  int(r.cycle)])
  return "\n".join([
      "## Pareto Frontier & Hardware Trade-offs", "",
      f"**{len(df)}** scored attempts across **{df.run.nunique()}** runs on"
      f" dataset `{this_fp}` (non-dominated accuracy-vs-size frontier):", "",
      _md_table([
          "KB", "val F1", "MFLOPs", "arena KB", "val score",
          "run", "student", "cycle",
      ], table), "",
      _caption("pareto", "Multi-objective Pareto dashboard: (a) validation F1"
               " vs model KB, (b) validation F1 vs MFLOPs, (c) activation"
               " arena RAM vs model flash KB colored by F1, and (d) overall"
               " score vs model KB showing the 8–14.5 KB sweet spot."), "",
  ])


def _model_family(student_name: str) -> str:
  """Canonical LLM family name from a student identifier."""
  s = re.sub(r"^student_\d+_", "", str(student_name))
  s = re.sub(r"_r\d+$", "", s)
  known = {
      "gemini_4_argon": "gemini-4-argon",
      "gemini_3.5_flash": "gemini-3.5-flash",
      "gemini_3.1_pro_preview": "gemini-3.1-pro-preview",
      "gemini_3.5_flash_lite": "gemini-3.5-flash-lite",
  }
  return known.get(s, s.replace("_", "-"))


def update_leaderboard_elo(lb_path: str | None = None) -> dict[str, int]:
  """Computes and persists a pairwise head-to-head `elo` column in leaderboard.tsv."""
  path = lb_path or os.path.join(TASK_DIR, "leaderboard.tsv")
  if not os.path.isfile(path):
    return {}
  try:
    df = pd.read_csv(path, sep="\t")
  except Exception:  # pylint: disable=broad-exception-caught
    return {}
  if df.empty or "test_overall" not in df.columns:
    return {}
  elos = np.full(len(df), 1500.0, dtype=float)
  is_mock = (
      df["tags"].fillna("").astype(str) == "mock"
      if "tags" in df.columns
      else np.zeros(len(df), dtype=bool)
  )
  for _, grp in df[~is_mock].groupby("dataset", sort=False):
    idxs = list(grp.index)
    if len(idxs) < 2:
      continue
    for _ in range(3):
      deltas = np.zeros(len(df), dtype=float)
      for i_pos, idx_a in enumerate(idxs):
        for idx_b in idxs[i_pos + 1:]:
          sa = float(df.loc[idx_a, "test_overall"])
          sb = float(df.loc[idx_b, "test_overall"])
          if not (np.isfinite(sa) and np.isfinite(sb)):
            continue
          ea = 1.0 / (1.0 + 10.0 ** ((elos[idx_b] - elos[idx_a]) / 400.0))
          actual_a = 1.0 / (1.0 + 10.0 ** (-(sa - sb) / 0.02))
          d = 20.0 * (actual_a - ea)
          deltas[idx_a] += d
          deltas[idx_b] -= d
      elos += deltas
  df["elo"] = np.round(elos).astype(int)
  cols = list(df.columns)
  if "elo" in cols and "test_overall" in cols:
    cols.remove("elo")
    ins_at = cols.index("test_overall") + 1
    cols.insert(ins_at, "elo")
    df = df[cols]
  df.to_csv(path, sep="\t", index=False)
  if "solution_id" in df.columns:
    return {
        str(r["solution_id"]): int(r["elo"])
        for _, r in df.iterrows()
        if pd.notna(r.get("solution_id"))
    }
  return {}


def elo_section(run_dir: str, summary: Mapping[str, Any],
                figs: str) -> tuple[str, dict[str, Any]]:
  """Pairwise Elo ratings across student LLM families and runs."""
  base = os.path.dirname(os.path.dirname(run_dir))
  this_fp = summary["dataset_fingerprint"]
  update_leaderboard_elo()

  # Discover all non-mock runs on this dataset in chronological order
  run_dirs = []
  for r_cand in sorted(glob.glob(os.path.join(base, "*", "*"))):
    if not os.path.isdir(os.path.join(r_cand, "students")):
      continue
    if os.path.isfile(os.path.join(r_cand, "MOCK")):
      continue
    log_p = os.path.join(r_cand, "run.log")
    fp = ""
    if os.path.isfile(log_p):
      with open(log_p, encoding="utf-8", errors="ignore") as lf:
        for ln in lf:
          m = re.search(r"dataset (\w+):", ln)
          if m:
            fp = m.group(1)
            break
    if fp == this_fp:
      run_dirs.append(r_cand)
  if not run_dirs:
    return "", {}

  families = [
      "gemini-4-argon",
      "gemini-3.5-flash",
      "gemini-3.1-pro-preview",
      "gemini-3.5-flash-lite",
  ]
  ratings: dict[str, float] = {f: 1500.0 for f in families}
  history: list[dict[str, Any]] = [
      {"step": 0, "label": "start", **ratings}
  ]
  wins: dict[tuple[str, str], float] = {}
  matches: dict[tuple[str, str], int] = {}
  cycle_firsts: dict[str, int] = {f: 0 for f in families}
  test_wins: dict[str, int] = {f: 0 for f in families}
  best_val: dict[str, float] = {f: -float("inf") for f in families}
  best_test: dict[str, float] = {f: -float("inf") for f in families}
  kb_list: dict[str, list[float]] = {f: [] for f in families}
  run_boundaries: list[tuple[int, str]] = []
  step = 0

  for rd in run_dirs:
    r_name = os.path.basename(rd)
    run_boundaries.append((step, r_name))
    # Load per-cycle scores by family (taking best if respawned in same cycle)
    cyc_scores: dict[int, dict[str, float]] = {}
    for csv_p in sorted(glob.glob(os.path.join(rd, "students", "*.csv"))):
      fam = _model_family(os.path.basename(csv_p)[:-4])
      if fam not in ratings:
        ratings[fam] = 1500.0
        families.append(fam)
        cycle_firsts[fam] = 0
        test_wins[fam] = 0
        best_val[fam] = -float("inf")
        best_test[fam] = -float("inf")
        kb_list[fam] = []
      try:
        sdf = pd.read_csv(csv_p)
      except Exception:  # pylint: disable=broad-exception-caught
        continue
      for _, row in sdf.iterrows():
        try:
          c_i = int(row["cycle"])
          sc = float(row["overall_score"])
        except (ValueError, TypeError, KeyError):
          continue
        if np.isnan(sc):
          sc = -float("inf")
        prev = cyc_scores.setdefault(c_i, {}).get(fam, -float("inf"))
        if sc >= prev:
          cyc_scores[c_i][fam] = sc
        if np.isfinite(sc):
          best_val[fam] = max(best_val[fam], sc)
          kb_v = float(row.get("metrics_model_kb", float("nan")))
          if np.isfinite(kb_v):
            kb_list[fam].append(kb_v)

    for c_i in sorted(cyc_scores):
      f_map = cyc_scores[c_i]
      if len(f_map) < 2:
        continue
      finite_fams = [f for f, s in f_map.items() if np.isfinite(s)]
      if finite_fams:
        top_f = max(finite_fams, key=lambda f: f_map[f])
        cycle_firsts[top_f] = cycle_firsts.get(top_f, 0) + 1
      deltas = {f: 0.0 for f in ratings}
      f_keys = sorted(f_map.keys())
      for idx_a, fa in enumerate(f_keys):
        for fb in f_keys[idx_a + 1:]:
          sa, sb = f_map[fa], f_map[fb]
          if not np.isfinite(sa) and not np.isfinite(sb):
            continue
          if not np.isfinite(sa):
            act_a = 0.0
          elif not np.isfinite(sb):
            act_a = 1.0
          else:
            act_a = 1.0 / (1.0 + 10.0 ** (-(sa - sb) / 0.025))
          ea = 1.0 / (1.0 + 10.0 ** ((ratings[fb] - ratings[fa]) / 400.0))
          d = 12.0 * (act_a - ea)
          deltas[fa] += d
          deltas[fb] -= d
          wins[(fa, fb)] = wins.get((fa, fb), 0.0) + act_a
          wins[(fb, fa)] = wins.get((fb, fa), 0.0) + (1.0 - act_a)
          matches[(fa, fb)] = matches.get((fa, fb), 0) + 1
          matches[(fb, fa)] = matches.get((fb, fa), 0) + 1
      for f in ratings:
        ratings[f] += deltas[f]
      step += 1
      history.append({"step": step, "label": f"{r_name}:C{c_i}", **ratings})

    # Final evaluation head-to-head on frozen test set (K=24)
    fe_path = os.path.join(rd, "final_eval_results.json")
    if os.path.isfile(fe_path):
      try:
        fe_data = json.load(open(fe_path, encoding="utf-8"))
        te_map: dict[str, float] = {}
        for item in fe_data:
          fam = _model_family(item.get("student_name", ""))
          b_sc = float((item.get("best") or {}).get("overall_score", -float("inf")))
          f_sc = float((item.get("first") or {}).get("overall_score", -float("inf")))
          sc = max(b_sc if np.isfinite(b_sc) else -float("inf"),
                   f_sc if np.isfinite(f_sc) else -float("inf"))
          if np.isfinite(sc):
            te_map[fam] = max(te_map.get(fam, -float("inf")), sc)
            best_test[fam] = max(best_test.get(fam, -float("inf")), sc)
        if len(te_map) >= 2:
          top_te = max(te_map, key=lambda f: te_map[f])
          test_wins[top_te] = test_wins.get(top_te, 0) + 1
          deltas = {f: 0.0 for f in ratings}
          t_keys = sorted(te_map.keys())
          for idx_a, fa in enumerate(t_keys):
            for fb in t_keys[idx_a + 1:]:
              sa, sb = te_map[fa], te_map[fb]
              act_a = 1.0 / (1.0 + 10.0 ** (-(sa - sb) / 0.02))
              ea = 1.0 / (1.0 + 10.0 ** ((ratings[fb] - ratings[fa]) / 400.0))
              d = 24.0 * (act_a - ea)
              deltas[fa] += d
              deltas[fb] -= d
              wins[(fa, fb)] = wins.get((fa, fb), 0.0) + act_a
              wins[(fb, fa)] = wins.get((fb, fa), 0.0) + (1.0 - act_a)
              matches[(fa, fb)] = matches.get((fa, fb), 0) + 1
              matches[(fb, fa)] = matches.get((fb, fa), 0) + 1
          for f in ratings:
            ratings[f] += deltas[f]
          step += 1
          history.append({"step": step, "label": f"{r_name}:Test", **ratings})
      except Exception:  # pylint: disable=broad-exception-caught
        pass

  hist_df = pd.DataFrame(history)
  fig, axes_e = _subplots(2, 1, w=5.4, h=4.1)
  ax_tr = axes_e[0][0]
  for i, fam in enumerate(families):
    if fam in hist_df:
      ax_tr.plot(hist_df["step"], hist_df[fam], lw=1.8,
                 color=_PALETTE[i % len(_PALETTE)],
                 label=f"{fam} ({ratings[fam]:.0f})")
  ax_tr.axhline(1500.0, color="k", ls=":", lw=0.8)
  prev_lbl_step = -999
  for b_step, b_name in run_boundaries[1:]:
    ax_tr.axvline(b_step, color="#888888", ls="--", lw=0.7, alpha=0.6)
    if b_step - prev_lbl_step >= 3:
      ax_tr.text(b_step + 0.25, ax_tr.get_ylim()[0] + 5, b_name[-6:],
                 fontsize=7, rotation=90, va="bottom", color="#555555")
      prev_lbl_step = b_step
  ax_tr.set_xlabel("classroom match step (cycles + frozen test evaluations)")
  ax_tr.set_ylabel("Elo rating (start = 1500)")
  ax_tr.set_title("Student LLM Elo trajectory across cycles & runs")
  ax_tr.legend(loc="best", fontsize=8)

  # Panel (b): Pairwise head-to-head win rate matrix (%)
  ax_mat = axes_e[0][1]
  n_f = len(families)
  win_mat = np.full((n_f, n_f), np.nan)
  for i, fa in enumerate(families):
    for j, fb in enumerate(families):
      if i == j:
        continue
      m_cnt = matches.get((fa, fb), 0)
      if m_cnt > 0:
        win_mat[i, j] = 100.0 * wins.get((fa, fb), 0.0) / m_cnt
  im = ax_mat.imshow(win_mat, cmap="RdYlGn", vmin=20, vmax=80, aspect="auto")
  short_f = [f.replace("gemini-", "") for f in families]
  ax_mat.set_xticks(range(n_f))
  ax_mat.set_xticklabels(short_f, rotation=25, ha="right", fontsize=8)
  ax_mat.set_yticks(range(n_f))
  ax_mat.set_yticklabels(short_f, fontsize=8)
  for i in range(n_f):
    for j in range(n_f):
      if i == j:
        ax_mat.text(j, i, "—", ha="center", va="center", fontsize=9)
      elif np.isfinite(win_mat[i, j]):
        m_cnt = matches.get((families[i], families[j]), 0)
        ax_mat.text(j, i, f"{win_mat[i, j]:.0f}%\n({m_cnt})",
                    ha="center", va="center", fontsize=7.5,
                    color="black")
  ax_mat.set_xlabel("opponent model")
  ax_mat.set_ylabel("row model")
  ax_mat.set_title("Head-to-head expected win rate % (matches)")
  ax_mat.grid(False)
  _finish(fig, os.path.join(figs, "elo_ratings.png"))

  # Build summary table sorted by Elo rating
  ranked_fams = sorted(families, key=lambda f: ratings[f], reverse=True)
  elo_rows = []
  for fam in ranked_fams:
    tot_w = sum(wins.get((fam, fb), 0.0) for fb in families if fb != fam)
    tot_m = sum(matches.get((fam, fb), 0) for fb in families if fb != fam)
    wr = f"{100.0 * tot_w / tot_m:.1f}% ({tot_m})" if tot_m else "-"
    bv = f"{best_val[fam]:.4f}" if np.isfinite(best_val[fam]) else "-"
    bt = f"{best_test[fam]:.4f}" if np.isfinite(best_test[fam]) else "-"
    mkb = f"{np.mean(kb_list[fam]):.1f}" if kb_list[fam] else "-"
    elo_rows.append([
        fam,
        int(round(ratings[fam])),
        wr,
        cycle_firsts.get(fam, 0),
        test_wins.get(fam, 0),
        bv,
        bt,
        mkb,
    ])
  elo_summary = {
      "ratings": {f: round(ratings[f], 1) for f in ranked_fams},
      "steps": step,
      "runs": len(run_dirs),
  }
  text = "\n".join([
      "## Student LLM Elo Ratings & Head-to-Head Record", "",
      f"Pairwise smooth-margin Elo (`start=1500`, `K_cycle=12`, `K_test=24`)"
      f" across **{len(run_dirs)}** runs (**{step}** classroom rounds) on"
      f" dataset `{this_fp}`:", "",
      _md_table([
          "LLM family", "Elo", "win rate (matches)", "cycle #1s",
          "test #1s", "best val", "best test", "mean KB",
      ], elo_rows), "",
      _caption("elo_ratings", "(a) LLM student Elo trajectory across all"
               " classroom cycles and frozen-test evaluations; (b) pairwise"
               " head-to-head win rate matrix."), "",
  ])
  return text, elo_summary


def _lineage_dir(students_dir: str) -> str:
  """Temp copy of students/ with `<name>_r<N>.csv` respawns merged by name."""
  out = tempfile.mkdtemp(prefix="lineage_")
  groups: dict[str, list[pd.DataFrame]] = {}
  for f in sorted(glob.glob(os.path.join(students_dir, "*.csv"))):
    base = re.sub(r"_r\d+$", "", os.path.basename(f)[:-4])
    groups.setdefault(base, []).append(pd.read_csv(f))
  for base, dfs in groups.items():
    df = pd.concat(dfs, ignore_index=True)
    if "cycle" in df:
      df = df.sort_values("cycle", kind="stable")
    df.to_csv(os.path.join(out, f"{base}.csv"), index=False)
  return out


def progression_section(run_dir: str, figs: str) -> str:
  """Reuses the library's progress grid and final-test bars."""
  from codegen_classroom import plotting  # pylint: disable=import-outside-toplevel
  students_dir = os.path.join(run_dir, "students")
  files = sorted(glob.glob(os.path.join(students_dir, "*.csv")))
  if not files:
    return ""
  rows = []
  for f in files:
    df = pd.read_csv(f)
    df = df[pd.to_numeric(df["overall_score"], errors="coerce").notna()]
    if df.empty:
      continue
    best = df.loc[df["overall_score"].idxmax()]
    rows.append([os.path.basename(f)[:-4].replace("student_", ""),
                 int(best["cycle"]), float(best["overall_score"]),
                 float(best.get("metrics_f1_macro", float("nan"))),
                 float(best.get("metrics_model_kb", float("nan"))),
                 float(best.get("metrics_mflops", float("nan"))), len(df)])
  parts = [
      "## Classroom Progression", "",
      _md_table(["student", "best cycle", "val score", "val F1", "KB",
                 "MFLOPs", "scored cycles"], rows), "",
  ]
  try:
    plotting.plot_progress_static(
        _lineage_dir(students_dir),
        metric_names=["overall_score", "f1_macro", "model_kb",
                      "mflops", "null_fpr", "worst_class_recall"],
        save_fig=os.path.join(figs, "progress.png"), show=False, dpi=DPI,
        single_plot_size=(4.6, 2.7))
    parts += [_caption("progress", "Per-student learning trajectory across"
                       " cycles (lines = best-so-far, diamonds = peer"
                       " collaboration)."), ""]
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning("progress plot failed: %s", e)
  final = os.path.join(run_dir, "final_eval_results.json")
  if os.path.isfile(final):
    try:
      plotting.plot_final_test(
          metrics=["overall_score", "f1_macro", "model_kb", "mflops"],
          final_results_path=final, backend="matplotlib",
          save_path=os.path.join(figs, "final_test.png"), show=False)
      parts += [_caption("final_test", "Frozen-test metrics per student."), ""]
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("final test plot failed: %s", e)
  return "\n".join(parts)


def leaderboard_section(figs: str) -> str:
  """Leaderboard history as a table (with Elo rating column)."""
  del figs
  path = os.path.join(TASK_DIR, "leaderboard.tsv")
  if not os.path.isfile(path):
    return ""
  update_leaderboard_elo(path)
  df = pd.read_csv(path, sep="\t")
  if "tags" in df:
    df = df[df["tags"].fillna("").astype(str) != "mock"]
  if df.empty:
    return ""
  cols = [c for c in ("student", "test_overall", "elo", "test_f1_macro",
                      "test_wcr", "model_kb", "mflops", "arena_kb",
                      "retrain_delta", "solution_id", "dataset", "timestamp")
          if c in df]
  return "\n".join(["## Leaderboard History", "",
                     _md_table(cols, df[cols].tail(10).values.tolist()), ""])


# ---------------------------------------------------------------------------
# Hardware in the loop
# ---------------------------------------------------------------------------


def _find_port() -> str | None:
  ports = sorted(glob.glob("/dev/cu.usbmodem*"))
  return ports[0] if ports else None


def _port_free(port: str) -> bool:
  import serial  # pylint: disable=import-outside-toplevel
  try:
    serial.Serial(port, 115200, timeout=0.1).close()
    return True
  except Exception:  # pylint: disable=broad-exception-caught
    return False


def hil_section(export_dir: str, splits: data_loader.Splits,
                host_eval: Mapping[str, Any], promoted: bool,
                log_path: str) -> tuple[str, dict[str, Any] | None]:
  """Flashes the exported model and replays the test split on the board."""
  import serial  # pylint: disable=import-outside-toplevel
  port = _find_port()
  if not port:
    return "## Hardware in the Loop\n\nSkipped: no `/dev/cu.usbmodem*` port.\n", None
  if not _port_free(port):
    return ("## Hardware in the Loop\n\nSkipped: serial port busy (disconnect"
            " in Playground tab and run `make device-eval`).\n"), None

  header = os.path.join(export_dir, "model_data.h")
  backup = None
  if not promoted and os.path.isfile(FIRMWARE_HEADER):
    backup = FIRMWARE_HEADER + ".bak"
    shutil.copy(FIRMWARE_HEADER, backup)
  shutil.copy(header, FIRMWARE_HEADER)
  t0 = time.time()
  with open(log_path, "w", encoding="utf-8") as lf:
    rc = subprocess.call(["make", "flash-codegen", f"PORT={port}"],
                         cwd=REPO_DIR, stdout=lf, stderr=subprocess.STDOUT)
  flash_s = time.time() - t0
  if backup:
    shutil.move(backup, FIRMWARE_HEADER)
  if rc != 0:
    return (f"## Hardware in the Loop\n\nFlash failed (exit {rc}); see"
            f" `{os.path.basename(log_path)}`.\n"), None

  # Firmware size from the arduino-cli summary line.
  flash_bytes = ram_bytes = None
  with open(log_path, encoding="utf-8") as lf:
    for line in lf:
      if line.startswith("Sketch uses"):
        flash_bytes = int(line.split()[2])
      elif line.startswith("Global variables use"):
        ram_bytes = int(line.split()[3])

  feats = data_loader.features_only(splits.x_test).reshape(
      len(splits.x_test), -1, data_loader.NUM_AXES)
  truth = splits.y_test.iloc[:, 0].to_numpy(int)
  host_pred = None
  preds, lats = [], []
  info = arena = None
  deadline = time.time() + 20
  while time.time() < deadline:
    try:
      with serial.Serial(port, 115200, timeout=0.5) as ser:
        time.sleep(1.0)
        ser.write(b"STATUS\n")
        info = deploy._read_until(ser, "INFO:", 5.0)  # pylint: disable=protected-access
        arena = int(deploy._read_until(ser, "ARENA:", 5.0)[6:])  # pylint: disable=protected-access
        for w in feats:
          p, lat = deploy.replay_window(ser, w)
          preds.append(int(np.argmax(p)))
          lats.append(lat)
        ser.write(b"LIVE\n")
      break
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.info("HIL retry: %s", e)
      preds, lats = [], []
      time.sleep(2.0)
  if not preds:
    return "## Hardware in the Loop\n\nFlashed, but the board did not answer REPLAY.\n", None

  from sklearn import metrics as skm  # pylint: disable=import-outside-toplevel
  preds_arr = np.asarray(preds)
  model_bytes = open(os.path.join(export_dir, "model.tflite"), "rb").read()
  host_pred = deploy.run_tflite(model_bytes, feats.reshape(len(feats), -1)).argmax(1)
  host_acc = float((host_pred == truth).mean())
  host_f1 = float(skm.f1_score(truth, host_pred, average="macro",
                               zero_division=0))
  out = {
      "port": port, "flash_seconds": flash_s, "firmware_flash_bytes": flash_bytes,
      "firmware_static_ram_bytes": ram_bytes, "arena_bytes": arena,
      "host_accuracy": host_acc,
      "host_f1_macro": host_f1,
      "device_accuracy": float((preds_arr == truth).mean()),
      "device_f1_macro": float(skm.f1_score(truth, preds_arr, average="macro",
                                            zero_division=0)),
      "device_latency_ms_p50": float(np.median(lats) / 1000.0),
      "device_latency_ms_max": float(np.max(lats) / 1000.0),
      "host_device_agreement": float((preds_arr == host_pred).mean()),
      "info": info,
  }
  rows = [[
      host_acc,
      out["device_accuracy"],
      host_f1,
      out["device_f1_macro"],
      f"{out['device_latency_ms_p50']:.2f} (max {out['device_latency_ms_max']:.2f})",
      arena,
      out["host_device_agreement"],
  ]]
  text = "\n".join([
      "## Hardware in the Loop (XIAO nRF52840 Sense)", "",
      f"Flashed `Arduino/Codegen` on `{port}` ({flash_s:.0f} s) | Flash"
      f" `{flash_bytes} B`, Static RAM `{ram_bytes} B` | `{info}`.", "",
      _md_table([
          "host acc", "device acc", "host F1", "device F1",
          "device latency ms", "arena bytes", "host-device agreement",
      ], rows), "",
  ])
  return text, out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_report(run_dir: str, splits: data_loader.Splits,
                 hil: str = "auto", promoted: bool = False,
                 figs_dir: str | None = None, fig_prefix: str = "",
                 fig_rel: str = "figs") -> str:
  """Writes export/REPORT.md and returns its path.

  With `figs_dir`/`fig_prefix`/`fig_rel`, figures are also copied as
  `<figs_dir>/<fig_prefix><name>.png` and the markdown references rewritten
  to `<fig_rel>/<fig_prefix><name>.png` (used for solutions/plots/).
  """
  _fig_counter["n"] = 0
  export_dir = os.path.join(run_dir, "export")
  figs = os.path.join(export_dir, "figs")
  os.makedirs(figs, exist_ok=True)
  summary = json.load(open(os.path.join(export_dir, "summary.json"),
                           encoding="utf-8"))
  model_bytes = open(os.path.join(export_dir, "model.tflite"), "rb").read()
  artifact = summary["artifact"]
  sections: list[str] = []
  extras: dict[str, Any] = {}

  tm = summary["test_metrics"]
  head = [
      f"# Micro-Gesture Solution Report — Run `{os.path.basename(run_dir)}`", "",
      f"- **Dataset**: `{summary['dataset_fingerprint']}`"
      f" ({len(summary['class_names'])} classes:"
      f" `{', '.join(summary['class_names'])}`)",
      f"- **Generated**: {pd.Timestamp.now().isoformat(timespec='seconds')} |"
      " **Exported**: `model.tflite`, `model_data.h`, `solution.py`,"
      " `summary.json`, `analysis.md`", "",
  ]
  sections.append("\n".join(head))

  qual_headers = [
      "overall score", "macro F1", "balanced acc", "worst recall",
      "Null FPR", "TFLite agree", "retrain delta",
  ]
  qual_row = [[
      summary["test_overall_score"],
      tm.get("f1_macro", float("nan")),
      tm.get("balanced_accuracy", float("nan")),
      tm.get("worst_class_recall", float("nan")),
      tm.get("null_fpr", float("nan")),
      tm.get("tflite_agreement", float("nan")),
      summary.get("retrain_delta", 0.0),
  ]]
  cost_headers = [
      "model KB", "arena KB (est)", "MFLOPs", "params (k)",
      "MCU ms (est)", "host ms", "int8 frac",
  ]
  cost_row = [[
      artifact["model_kb"], artifact["arena_kb"], artifact["mflops"],
      artifact["params_k"], artifact["mcu_ms_est"],
      artifact["host_latency_ms"], artifact["int8_fraction"],
  ]]
  sections.append("\n".join([
      "## Scorecard (Frozen Test & Footprint)", "",
      _md_table(qual_headers, qual_row), "",
      _md_table(cost_headers, cost_row), "",
  ]))

  def run(name, fn, *a, **k):
    try:
      return fn(*a, **k)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("report section %s failed: %s", name, e, exc_info=True)
      return f"## {name}\n\nSection failed: {e}\n"

  analysis_md = os.path.join(export_dir, "analysis.md")
  if os.path.isfile(analysis_md):
    body = open(analysis_md, encoding="utf-8").read()
    body = body.split("\n", 1)[1] if body.startswith("# ") else body
    sections.append("## Winning Solution Analysis\n" + body)

  sections.append(run("Dataset", dataset_section, splits, figs))
  sections.append(run("Model", model_section, model_bytes, artifact, figs,
                      splits))
  ev = run("Evaluation", evaluation_section, model_bytes, splits, figs)
  if isinstance(ev, tuple):
    sections.append(ev[0])
    extras["evaluation"] = ev[1]
  else:
    sections.append(ev)
  cv = run("Classifier analysis", curves_section, model_bytes, splits, figs)
  if isinstance(cv, tuple):
    sections.append(cv[0])
    extras["curves"] = cv[1]
  else:
    sections.append(cv)
  st = run("Streaming simulation", stream_section, model_bytes, splits, figs)
  if isinstance(st, tuple):
    sections.append(st[0])
    extras["stream"] = st[1]
  else:
    sections.append(st)
  sections.append(run("Host benchmark", host_benchmark_section, model_bytes,
                      data_loader.features_only(splits.x_test), figs))
  if hil in ("auto", "on"):
    h = run("Hardware in the loop", hil_section, export_dir, splits,
            extras.get("evaluation", {}), promoted,
            os.path.join(export_dir, "flash.log"))
    if isinstance(h, tuple):
      sections.append(h[0])
      if h[1]:
        extras["device"] = h[1]
    else:
      sections.append(h)
  elif isinstance(summary.get("report", {}).get("device"), dict):
    dev = summary["report"]["device"]
    extras["device"] = dev
    host_t = extras.get("evaluation", {}).get("test", {})
    rows = [[
        host_t.get("accuracy", float("nan")),
        dev.get("device_accuracy", float("nan")),
        host_t.get("f1_macro", float("nan")),
        dev.get("device_f1_macro", float("nan")),
        f"{dev.get('device_latency_ms_p50', 0):.2f}"
        f" (max {dev.get('device_latency_ms_max', 0):.2f})",
        dev.get("arena_bytes", "-"),
        dev.get("host_device_agreement", float("nan")),
    ]]
    sections.append("\n".join([
        "## Hardware in the Loop (XIAO nRF52840 Sense)", "",
        f"Flashed `Arduino/Codegen` on `{dev.get('port', '')}`"
        f" ({dev.get('flash_seconds', 0):.0f} s) | Flash"
        f" `{dev.get('firmware_flash_bytes', '')} B`, Static RAM"
        f" `{dev.get('firmware_static_ram_bytes', '')} B` |"
        f" `{dev.get('info', '')}`.", "",
        _md_table([
            "host acc", "device acc", "host F1", "device F1",
            "device latency ms", "arena bytes", "host-device agreement",
        ], rows), "",
    ]))
  sections.append(run("Pareto frontier", pareto_section, run_dir, summary,
                      figs))
  el = run("Elo ratings", elo_section, run_dir, summary, figs)
  if isinstance(el, tuple):
    sections.append(el[0])
    if el[1]:
      extras["elo"] = el[1]
  else:
    sections.append(el)
  sections.append(run("Classroom progression", progression_section, run_dir,
                      figs))
  sections.append(run("Leaderboard history", leaderboard_section, figs))

  text = "\n".join(s for s in sections if s)
  with open(os.path.join(export_dir, "REPORT.md"), "w", encoding="utf-8") as f:
    f.write(text)
  if figs_dir:
    os.makedirs(figs_dir, exist_ok=True)
    for name in os.listdir(figs):
      if name.endswith(".png"):
        shutil.copy(os.path.join(figs, name),
                    os.path.join(figs_dir, f"{fig_prefix}{name}"))
    text = re.sub(r"\(figs/([^)]+\.png)\)",
                  lambda m: f"({fig_rel}/{fig_prefix}{m.group(1)})", text)
  build_report.last_text = text  # type: ignore[attr-defined]
  if extras:
    summary["report"] = extras
    with open(os.path.join(export_dir, "summary.json"), "w",
              encoding="utf-8") as f:
      json.dump(summary, f, indent=2)
  return os.path.join(export_dir, "REPORT.md")


def analysis_markdown() -> str:
  """Markdown of the most recent build_report() with rewritten figure paths."""
  return getattr(build_report, "last_text", "")


def latest_run_dir() -> str | None:
  base = os.environ.get("CODEGEN_EXPERIMENTS_DIR",
                        os.path.join(REPO_DIR, "experiments"))
  runs = sorted(glob.glob(os.path.join(base, "micro_gesture", "*", "*",
                                       "export", "summary.json")))
  return os.path.dirname(os.path.dirname(runs[-1])) if runs else None


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--run_dir", default="latest")
  ap.add_argument("--hil", default="auto", choices=["auto", "on", "off"])
  ap.add_argument("--data_dir", default=None)
  args = ap.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  run_dir = latest_run_dir() if args.run_dir == "latest" else args.run_dir
  if not run_dir:
    raise SystemExit("no run with an export found")
  splits = data_loader.load_splits(args.data_dir)
  deploy.configure(splits.class_names)
  path = build_report(run_dir, splits, hil=args.hil)
  print(path)


if __name__ == "__main__":
  main()
