"""Codegen tab: run, watch, and deploy the micro_gesture codegen task from the GUI.

Drives `python -m codegen_task.run_micro_gesture` (and `--watch`, `deploy`,
`make flash-codegen`) as child processes so the PyQt loop stays responsive, and
shows the dataset summary, streaming log, live cycle trajectory, student &
leaderboard bar plots, leaderboard, and rendered solution reports.
"""

from __future__ import annotations

import glob
import html
import json
import os
import re
import sys

from PyQt5.QtCore import QProcess, QProcessEnvironment, QTimer, Qt, QUrl
from PyQt5.QtGui import (
    QColor, QFont, QImage, QSyntaxHighlighter, QTextCharFormat, QTextCursor,
    QTextDocument,
)
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPlainTextEdit, QPushButton, QSpinBox, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget,
)
import numpy as np
import pandas as pd
import pyqtgraph as pg

from app_styles import AppStyles

PY_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PY_DIR)
TASK_DIR = os.path.join(PY_DIR, "codegen_task")
SOLUTIONS_DIR = os.path.join(TASK_DIR, "solutions")
EXP_DIR = os.path.join(REPO_DIR, "experiments")
LEADERBOARD = os.path.join(TASK_DIR, "leaderboard.tsv")
DEPLOYED_JSON = os.path.join(REPO_DIR, "Arduino", "Codegen", "deployed.json")
DEPLOYED_FALLBACK_JSON = os.path.join(EXP_DIR, "deployed", "deployed.json")
DEFAULT_MODELS = (
    "gemini-4-argon,gemini-3.5-flash,gemini-3.1-pro-preview,"
    "gemini-3.5-flash-lite"
)

# Log lines from TensorFlow, Keras, converter, and HTTP transport that add
# noise in Standard / Debug modes (still visible in Raw unfiltered mode).
_NOISE = (
    "I0000", "W0000", "oneDNN", "rebuild TensorFlow", "arith.constant",
    "(f32:", "(i32:", "Accepted dialects", "Non-Converted Ops", "ARITH ops",
    "fully_quantize", "Summary on the non-converted", "XNNPACK",
    "Ignored output_format", "Ignored drop_control", "SavedModel", "MLIR",
    "HTTP Request:", "AFC is enabled", "----------",
    "optimizer.py:70", "runs slowly on M1/M2 Macs",
    "polymorphic_function.py:", "triggered tf.function retracing",
    "save.py:289", "Found untraced functions",
    "builder_impl.py:801", "Assets written to:",
    "fingerprinting_utils.py:49", "Writing fingerprint to",
    "lite.py:1364", "Using new converter:",
    "tpu_cluster_resolver.py:", "signature_serialization.py:",
    "_trace.py:47", "_client.py:1025", "Creating converter from",
    "f32:", "i32:", "i8:",
    "font_manager.py:", "findfont:", "PngImagePlugin.py:", "STREAM b'",
    "CV folds predicted ",
)

_SIGNAL_PATTERNS = (
    "=== Cycle", "[cycle ", "Score:", "New best", "Excluding", "Respawn",
    "Best on test", "Promoted", "Keeping", "Retrain", "[retrain ok]",
    "Re-fitting", "Exported ", "Auto-Watch", "Analysis",
    "Report:", "Hardware", "Export:", "deploy ok:", "Deploy contract",
    "Error", "WARNING", "Traceback", "dataset ", "run_dir ", "Seeded ",
    "Cycle scoring", "Refining solution", "Mutating with", "$ ", "[exit",
)

_CYCLE_METRIC_RE = re.compile(
    r"\[cycle\s+(\d+)\]\s+(\S+):\s+overall=([0-9.]+)\s+f1=([0-9.]+)"
    r"\s+wcr=([0-9.]+)(?:\s+nfpr=([0-9.]+)\s+kb=([0-9.]+)\s+mflops=([0-9.]+)"
    r"\s+arena=([0-9.]+))?"
)
_LIB_SCORE_LINE_RE = re.compile(
    r"\[(student_[^\]]+)\]\s+Score:\s*([0-9.]+),\s*Metrics:\s*(.+?)"
    r"\s*\(Cycle:\s*(\d+)"
)
_KV_FLOAT_RE = re.compile(r"([a-zA-Z0-9_]+):\s*([0-9.]+)")
_MUTATE_PEER_RE = re.compile(
    r"\[(student_[^\]]+)\]\s+Mutating with.*?\(Peer:\s*([^)]+)\)\s*for cycle\s*(\d+)"
)
_RUN_DIR_RE = re.compile(r"\brun_dir\s+(\S+)")

# Bounded [0, 1] metrics for the top hill-climb plot:
# (key, label, color, default_checked, lower_is_better)
_BOUNDED_METRICS = (
    ("overall_score", "Overall", "#4ADE80", True, False),
    ("f1_macro", "F1 Macro", "#38BDF8", True, False),
    ("balanced_accuracy", "Bal Acc", "#A78BFA", True, False),
    ("worst_class_recall", "Worst Recall", "#F59E0B", True, False),
    ("null_fpr", "Null FPR", "#F43F5E", False, True),
    ("int8_fraction", "INT8 Frac", "#2DD4BF", False, False),
    ("tflite_agreement", "Agreement", "#E879F9", False, False),
)

# Modes selectable in the bottom bar chart (per-student stats + leaderboard histograms):
# (key, label, y_axis_label)
_STUDENT_BAR_MODES = (
    ("score_best_mean", "Students: Best vs Mean Score (0..1)", "Score"),
    ("f1_best_mean", "Students: Best vs Mean F1 Macro (0..1)", "F1 Macro"),
    ("wcr_best_mean", "Students: Best vs Mean Worst Recall (0..1)", "Worst Recall"),
    ("avg_rank", "Students: Average Cycle Rank (1 = 1st)", "Avg Rank"),
    ("code_loc", "Students: Solution Size (Lines of Code)", "Lines of Code"),
    ("model_kb", "Students: Best Model Flash Size (KB)", "Model KB"),
    ("mflops", "Students: Best Compute (MFLOPs)", "MFLOPs"),
    ("cycles_collab", "Students: Scored Cycles & Peer Collabs", "Cycles"),
    ("top_tags", "Leaderboard: Top Technique Tags (Histogram)", "Solutions"),
    ("top_ops", "Leaderboard: Top TFLite Ops (Histogram)", "Solutions"),
)

_STUDENT_COLORS = (
    "#38BDF8",  # sky blue
    "#4ADE80",  # green
    "#F59E0B",  # amber
    "#E879F9",  # magenta
    "#A78BFA",  # violet
    "#2DD4BF",  # teal
    "#FB7185",  # coral
    "#FACC15",  # yellow
)


def _fmt(color: str, bold: bool = False, italic: bool = False) -> QTextCharFormat:
  f = QTextCharFormat()
  f.setForeground(QColor(color))
  if bold:
    f.setFontWeight(QFont.Bold)
  if italic:
    f.setFontItalic(True)
  return f


class CodegenLogHighlighter(QSyntaxHighlighter):
  """Subtle dark-slate syntax and log highlighter for the codegen terminal."""

  def __init__(self, document):
    super().__init__(document)
    self.fmt_dim = _fmt("#64748B")
    self.fmt_cycle = _fmt("#7DD3FC", bold=True)
    self.fmt_student = _fmt("#C4B5FD", bold=True)
    self.fmt_good = _fmt("#86EFAC", bold=True)
    self.fmt_metric_val = _fmt("#A7F3D0")
    self.fmt_warn = _fmt("#FCD34D")
    self.fmt_err = _fmt("#FCA5A5", bold=True)
    self.fmt_kw = _fmt("#93C5FD", bold=True)
    self.fmt_comment = _fmt("#6EE7B7", italic=True)
    self.fmt_md_bold = _fmt("#E2E8F0", bold=True)

    self.rules: list[tuple[re.Pattern[str], QTextCharFormat]] = [
        # Timestamps and absl prefixes (dimmed)
        (re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d+\s+\w+"),
         self.fmt_dim),
        (re.compile(r"^[IWE]\d{4}\s+\d{2}:\d{2}:\d{2}\.\d+\s+\d+\s+\S+\]"),
         self.fmt_dim),
        (re.compile(r"\bcodegen_classroom(?:\.\w+)*:"), self.fmt_dim),
        # Cycle & section banners
        (re.compile(r"===\s*Cycle\s+\d+.*?==="), self.fmt_cycle),
        (re.compile(r"^\*\*.*\*\*:?$"), self.fmt_cycle),
        (re.compile(r"^#{1,3}\s+.*$"), self.fmt_cycle),
        (re.compile(r"^\$\s+.*$"), self.fmt_cycle),
        # Student identifiers
        (re.compile(r"\[student_[^\]]+\]"), self.fmt_student),
        (re.compile(r"\[cycle\s+\d+\]\s+\S+:"), self.fmt_student),
        # Positive milestones & key metrics
        (re.compile(
            r"\b(?:Score|overall|overall_score|f1_macro|f1|Best on test|"
            r"New best score|deploy ok|Promoted|Run frontier)\b[:=]?\s*[0-9.+->]*"),
         self.fmt_good),
        (re.compile(
            r"\b(?:worst_class_recall|wcr|balanced_accuracy|null_fpr|nfpr|"
            r"model_kb|kb|mflops|arena_kb|arena|int8_fraction|tflite_agreement)"
            r"[:=]\s*[0-9.]+"),
         self.fmt_metric_val),
        # Refinements, mutations, hypotheses & plans
        (re.compile(
            r"\b(?:Refining solution|Early stop|Mutating with|"
            r"CONTRASTIVE_PEER_CROSSOVER|BEST_PEER_CROSSOVER|SELF_BEST_CODE|"
            r"Seeded|Respawn\w*|Hyp|Hypothesis|plan)\b:?"),
         self.fmt_warn),
        # Warnings, errors, validator failures
        (re.compile(
            r"\b(?:WARNING|ERROR|ValidatorError|ExecutionError|FAILED|"
            r"Traceback|Excluding)\b.*"),
         self.fmt_err),
        # Python / Markdown snippets in debug mode
        (re.compile(
            r"\b(?:def|return|import|from|class|for|if|else|elif|try|except|"
            r"with|lambda)\b"),
         self.fmt_kw),
        (re.compile(r"\*\*[^*]+\*\*"), self.fmt_md_bold),
        (re.compile(r"^\s*#.*$"), self.fmt_comment),
        (re.compile(r"^```.*$"), self.fmt_comment),
    ]

  def highlightBlock(self, text: str) -> None:  # pylint: disable=invalid-name
    for pattern, fmt in self.rules:
      for m in pattern.finditer(text):
        self.setFormat(m.start(), m.end() - m.start(), fmt)


_LOG_HTML_STYLES: dict[str, str] = {
    "comment": "color:#6EE7B7;font-style:italic;",
    "cycle": "color:#7DD3FC;font-weight:bold;",
    "err": "color:#FCA5A5;font-weight:bold;",
    "dim": "color:#64748B;",
    "student": "color:#C4B5FD;font-weight:bold;",
    "good": "color:#86EFAC;font-weight:bold;",
    "metric": "color:#A7F3D0;",
    "warn": "color:#FCD34D;",
    "md_bold": "color:#E2E8F0;font-weight:bold;",
    "kw": "color:#93C5FD;font-weight:bold;",
}

_LOG_HTML_MASTER_RE = re.compile(
    r"(?P<comment>^\s*#.*$|^```.*$)"
    r"|(?P<cycle>===\s*Cycle\s+\d+.*?===|^\*\*.*\*\*:?$|^#{1,3}\s+.*$|^\$\s+.*$)"
    r"|(?P<err>\b(?:WARNING|ERROR|ValidatorError|ExecutionError|FAILED|Traceback|Excluding)\b.*)"
    r"|(?P<dim>^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d+\s+\w+"
    r"|^[IWE]\d{4}\s+\d{2}:\d{2}:\d{2}\.\d+\s+\d+\s+\S+\]"
    r"|\bcodegen_classroom(?:\.\w+)*:)"
    r"|(?P<student>\[student_[^\]]+\]|\[cycle\s+\d+\]\s+\S+:)"
    r"|(?P<good>\b(?:Score|overall|overall_score|f1_macro|f1|Best on test|"
    r"New best score|deploy ok|Promoted|Run frontier)\b[:=]?\s*[0-9.+->]*)"
    r"|(?P<metric>\b(?:worst_class_recall|wcr|balanced_accuracy|null_fpr|nfpr|"
    r"model_kb|kb|mflops|arena_kb|arena|int8_fraction|tflite_agreement)[:=]\s*[0-9.]+)"
    r"|(?P<warn>\b(?:Refining solution|Early stop|Mutating with|"
    r"CONTRASTIVE_PEER_CROSSOVER|BEST_PEER_CROSSOVER|SELF_BEST_CODE|"
    r"Seeded|Respawn\w*|Hyp|Hypothesis|plan)\b:?)"
    r"|(?P<md_bold>\*\*[^*]+\*\*)"
    r"|(?P<kw>\b(?:def|return|import|from|class|for|if|else|elif|try|except|"
    r"with|lambda)\b)"
)

_LOG_LEVEL_LINE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d+\s+(DEBUG|INFO|WARNING|ERROR)\b"
)


def _colorize_log_line_html(line: str) -> str:
  """Formats a single log line with ClassroomLogHighlighter HTML spans."""
  parts: list[str] = []
  last_end = 0
  for m in _LOG_HTML_MASTER_RE.finditer(line):
    if m.start() > last_end:
      parts.append(html.escape(line[last_end:m.start()]))
    grp = m.lastgroup or ""
    style = _LOG_HTML_STYLES.get(grp, "")
    token = html.escape(m.group(0))
    if style:
      parts.append(f'<span style="{style}">{token}</span>')
    else:
      parts.append(token)
    last_end = m.end()
  if last_end < len(line):
    parts.append(html.escape(line[last_end:]))
  return "".join(parts)


def _invert_plot_for_dark_mode(qimg: QImage) -> QImage:
  """Hue-preserving lightness inversion mapping white plots to dark slate."""
  img = qimg.convertToFormat(QImage.Format_RGBA8888)
  w, h = img.width(), img.height()
  if w <= 0 or h <= 0:
    return qimg
  bpl = img.bytesPerLine()
  ptr = img.bits()
  ptr.setsize(h * bpl)
  buf = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl)).copy()
  arr = buf[:, :w * 4].reshape((h, w, 4))
  rgb = arr[:, :, :3].astype(np.float32) / 255.0
  c_max = rgb.max(axis=2, keepdims=True)
  c_min = rgb.min(axis=2, keepdims=True)
  # Hue-preserving HSL lightness inversion (L' = 1 - L)
  inv = np.clip(1.0 - c_max - c_min + rgb, 0.0, 1.0)
  bg = np.array([[[28.0, 32.0, 40.0]]], dtype=np.float32)
  fg = np.array([[[236.0, 239.0, 244.0]]], dtype=np.float32)
  arr[:, :, :3] = np.clip(bg + inv * (fg - bg), 0.0, 255.0).astype(np.uint8)
  out = QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888)
  return out.copy()


class CodegenWidget(QWidget):
  """GUI front end for the micro_gesture codegen task."""

  def __init__(self, serial_handler=None, parent=None):
    super().__init__(parent)
    self.serial_handler = serial_handler
    self.process: QProcess | None = None
    self.active_run_dir: str | None = None
    self._loaded_run_dir: str | None = None
    self._pinned_report_run_dir: str | None = None
    self._run_launch_time: float = 0.0
    # (cycle, student) -> dict of metric values + "collab": bool + "loc": int
    self._live_records: dict[tuple[int, str], dict[str, float | bool]] = {}
    self._collab_cycles: set[tuple[int, str]] = set()
    self._report_paths: list[tuple[str, str]] = []
    self._build_ui()
    self._poll_timer = QTimer(self)
    self._poll_timer.setInterval(2000)
    self._poll_timer.timeout.connect(self._poll_active_run)
    self.refresh()
    self._poll_timer.start()

  # -- UI ---------------------------------------------------------------------

  def _build_ui(self) -> None:
    root = QHBoxLayout(self)
    root.setSpacing(AppStyles.WIDGET_SPACING)

    left_widget = QWidget()
    left_widget.setFixedWidth(260)
    left = QVBoxLayout(left_widget)
    left.setContentsMargins(0, 0, 0, 0)
    left.setSpacing(8)
    left.addWidget(self._dataset_box())
    left.addWidget(self._run_box())
    left.addWidget(self._device_box())
    left.addStretch(1)
    root.addWidget(left_widget, 0)

    right_split = QSplitter(Qt.Vertical)

    # Top-right: Wide tabbed [Live Run Log | Solution] on left,
    # separate Real-Time Cycle Trajectory & Bar Plots on right
    top_right = QSplitter(Qt.Horizontal)
    self.log = QPlainTextEdit()
    self.log.setReadOnly(True)
    self.log.setMaximumBlockCount(8000)
    self.log.setStyleSheet(
        "background-color: #1E222A; color: #D8DEE9;"
        " font-family: Menlo, Monaco, monospace; font-size: 10px;"
        " padding: 6px; border-radius: 4px;"
    )
    self.highlighter = CodegenLogHighlighter(self.log.document())

    self.main_tabs = QTabWidget()
    self.main_tabs.addTab(self.log, "Live Run Log")
    self.main_tabs.addTab(self._report_viewer_box(), "Solution")
    self.main_tabs.currentChanged.connect(self._on_main_tab_changed)

    top_right.addWidget(self.main_tabs)
    top_right.addWidget(self._metrics_graph_box())
    top_right.setStretchFactor(0, 7)
    top_right.setStretchFactor(1, 3)
    top_right.setSizes([800, 320])
    right_split.addWidget(top_right)

    # Bottom-right: Full-width Leaderboard
    self.table = QTableWidget()
    self.table.setEditTriggers(QTableWidget.NoEditTriggers)
    self.table.setSelectionBehavior(QTableWidget.SelectRows)
    self.table.cellClicked.connect(self._on_leaderboard_row_clicked)
    right_split.addWidget(
        self._wrap(
            "Leaderboard (click any row to open its report in the viewer above)",
            self.table,
        )
    )
    right_split.setStretchFactor(0, 7)
    right_split.setStretchFactor(1, 3)
    right_split.setSizes([540, 200])

    root.addWidget(right_split, 1)

  def _wrap(self, title: str, widget: QWidget) -> QGroupBox:
    box = QGroupBox(title)
    box.setStyleSheet(AppStyles.get_group_box_style())
    lay = QVBoxLayout(box)
    lay.addWidget(widget)
    return box

  def _style_plot(self, plot: pg.PlotWidget, y_label: str,
                  x_label: str = "Cycle", bounded: bool = False) -> None:
    plot.setMinimumWidth(240)
    plot.setBackground("#232730")
    plot.showGrid(x=True, y=True, alpha=0.28)
    if bounded:
      plot.setYRange(-0.02, 1.05)
    label_style = {
        "color": AppStyles.GRAPH_LABEL_COLOR,
        "font-size": "9pt",
    }
    plot.getAxis("left").setLabel(y_label, **label_style)
    plot.getAxis("bottom").setLabel(x_label, **label_style)
    plot.getAxis("left").setTextPen("w")
    plot.getAxis("bottom").setTextPen("w")
    font = QFont()
    font.setPointSize(8)
    plot.getAxis("bottom").setStyle(tickFont=font)
    plot.getAxis("left").setStyle(tickFont=font)

  def _metrics_graph_box(self) -> QWidget:
    container = QWidget()
    outer = QVBoxLayout(container)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(6)

    # 1. Top GroupBox: Real-Time Cycle Trajectory (CV)
    top_panel = QWidget()
    top_lay = QVBoxLayout(top_panel)
    top_lay.setContentsMargins(2, 2, 2, 2)
    top_lay.setSpacing(4)

    toggles_grid = QGridLayout()
    toggles_grid.setContentsMargins(0, 0, 0, 0)
    toggles_grid.setHorizontalSpacing(6)
    toggles_grid.setVerticalSpacing(2)
    self._metric_checks: dict[str, QCheckBox] = {}
    self._metric_curves: dict[str, pg.PlotDataItem] = {}

    self.metrics_plot = pg.PlotWidget()
    self._style_plot(self.metrics_plot, "Best Score", x_label="Cycle",
                     bounded=False)

    symbols = ("o", "t", "s", "d", "p", "h", "+")
    for idx, (key, label, color, default_on, _low) in enumerate(_BOUNDED_METRICS):
      chk = QCheckBox(label)
      chk.setChecked(default_on)
      chk.setStyleSheet(
          f"QCheckBox {{ color: {color}; font-weight: bold; font-size: 10px; }}"
      )
      chk.toggled.connect(self._update_metrics_graph)
      self._metric_checks[key] = chk
      toggles_grid.addWidget(chk, idx // 4, idx % 4)

      curve = self.metrics_plot.plot(
          pen=pg.mkPen(color, width=2.5),
          symbol=symbols[idx % len(symbols)],
          symbolSize=6,
          symbolBrush=color,
          name=label,
      )
      curve.setVisible(default_on)
      self._metric_curves[key] = curve

    top_lay.addLayout(toggles_grid)
    top_lay.addWidget(self.metrics_plot, 1)
    self._metrics_box = self._wrap("Real-Time Cycle Trajectory (CV)", top_panel)
    outer.addWidget(self._metrics_box, 1)

    # 2. Bottom GroupBox: Student & Leaderboard Bar Plots
    bot_panel = QWidget()
    bot_lay = QVBoxLayout(bot_panel)
    bot_lay.setContentsMargins(2, 2, 2, 2)
    bot_lay.setSpacing(4)

    bot_row = QHBoxLayout()
    bot_row.setContentsMargins(0, 0, 0, 0)
    bot_lbl = QLabel("<b>Mode:</b>")
    bot_lbl.setStyleSheet("color: #E5E9F0; font-size: 11px;")
    bot_row.addWidget(bot_lbl)

    self.student_metric_combo = QComboBox()
    self.student_metric_combo.setStyleSheet("font-size: 10px;")
    for key, label, _ylbl in _STUDENT_BAR_MODES:
      self.student_metric_combo.addItem(label, key)
    self.student_metric_combo.currentIndexChanged.connect(
        self._update_bar_plot
    )
    bot_row.addWidget(self.student_metric_combo, 1)
    bot_lay.addLayout(bot_row)

    self.student_plot = pg.PlotWidget()
    self._style_plot(self.student_plot, "Score", x_label="Student",
                     bounded=False)
    bot_lay.addWidget(self.student_plot, 1)

    self._bar_box = self._wrap("Student & Leaderboard Bar Plots", bot_panel)
    outer.addWidget(self._bar_box, 1)
    return container

  def _report_viewer_box(self) -> QWidget:
    panel = QWidget()
    lay = QVBoxLayout(panel)
    lay.setContentsMargins(4, 4, 4, 4)
    top = QHBoxLayout()
    lbl = QLabel("Document:")
    lbl.setStyleSheet(AppStyles.get_label_style())
    top.addWidget(lbl)
    self.report_combo = QComboBox()
    self.report_combo.currentIndexChanged.connect(self._on_report_selected)
    top.addWidget(self.report_combo, 1)

    self.report_dark_mode = QCheckBox("Dark theme")
    self.report_dark_mode.setChecked(True)
    self.report_dark_mode.toggled.connect(self._apply_report_theme)
    top.addWidget(self.report_dark_mode)

    ext_btn = QPushButton("Open externally")
    ext_btn.setStyleSheet(AppStyles.get_button_style())
    ext_btn.clicked.connect(self.open_report_external)
    top.addWidget(ext_btn)
    lay.addLayout(top)

    self.report_browser = QTextBrowser()
    self.report_browser.setOpenExternalLinks(True)
    self._apply_report_theme()
    lay.addWidget(self.report_browser, 1)
    return panel

  def _apply_report_theme(self) -> None:
    if getattr(self, "report_dark_mode", None) and self.report_dark_mode.isChecked():
      self.report_browser.setStyleSheet(
          "background-color: #161920; color: #E5E9F0; padding: 12px;"
          " font-size: 13px; border-radius: 4px;"
      )
      self.report_browser.document().setDefaultStyleSheet(
          "body { color: #E5E9F0; background-color: #161920; line-height: 1.45; }"
          "h1, h2, h3, h4 { color: #88C0D0; margin-top: 14px; }"
          "a { color: #81A1C1; }"
          "code, pre { background-color: #232834; color: #A3BE8C;"
          " font-family: Menlo, Monaco, monospace; padding: 2px 4px; }"
          "table { border-collapse: collapse; margin: 8px 0; }"
          "th { background-color: #232834; color: #ECEFF4; padding: 5px 8px;"
          " border: 1px solid #3B4252; }"
          "td { padding: 4px 8px; border: 1px solid #2E3440; color: #D8DEE9; }"
      )
    else:
      self.report_browser.setStyleSheet(
          "background-color: #FFFFFF; color: #111827; padding: 12px;"
          " font-size: 13px; border-radius: 4px;"
      )
      self.report_browser.document().setDefaultStyleSheet(
          "body { color: #111827; background-color: #FFFFFF; line-height: 1.45; }"
          "h1, h2, h3, h4 { color: #1E3A8A; margin-top: 14px; }"
          "code, pre { background-color: #F3F4F6; color: #1F2937;"
          " font-family: Menlo, Monaco, monospace; padding: 2px 4px; }"
          "table { border-collapse: collapse; margin: 8px 0; }"
          "th { background-color: #E5E7EB; color: #111827; padding: 5px 8px;"
          " border: 1px solid #D1D5DB; }"
          "td { padding: 4px 8px; border: 1px solid #E5E7EB; color: #1F2937; }"
      )
    cur = self._current_report_path()
    if cur and os.path.isfile(cur):
      self._render_markdown_file(cur)

  def _dataset_box(self) -> QGroupBox:
    self.dataset_label = QLabel("")
    self.dataset_label.setStyleSheet(AppStyles.get_label_style())
    self.dataset_label.setWordWrap(True)
    self.deployed_label = QLabel("")
    self.deployed_label.setStyleSheet(AppStyles.get_label_style())
    self.deployed_label.setWordWrap(True)
    refresh = QPushButton("Refresh")
    refresh.setStyleSheet(AppStyles.get_button_style())
    refresh.clicked.connect(self.refresh)
    box = QGroupBox("Dataset and deployed model")
    box.setStyleSheet(AppStyles.get_group_box_style())
    lay = QVBoxLayout(box)
    lay.addWidget(self.dataset_label)
    lay.addWidget(self.deployed_label)
    lay.addWidget(refresh)
    return box

  def _run_box(self) -> QGroupBox:
    self.students_spin = QSpinBox()
    self.students_spin.setRange(1, 12)
    self.students_spin.setValue(4)

    self.cycles = QSpinBox()
    self.cycles.setRange(1, 30)
    self.cycles.setValue(8)

    self.cv_folds_spin = QSpinBox()
    self.cv_folds_spin.setRange(0, 10)
    self.cv_folds_spin.setValue(3)
    self.cv_folds_spin.setToolTip(
        "Stratified K-fold OOF scoring per cycle (0 = single hold-out split)"
    )

    self.timeout_spin = QSpinBox()
    self.timeout_spin.setRange(30, 600)
    self.timeout_spin.setSingleStep(15)
    self.timeout_spin.setValue(150)
    self.timeout_spin.setSuffix(" s")

    self.models = QLineEdit(DEFAULT_MODELS)
    self.models.setStyleSheet(AppStyles.get_line_edit_style())
    self.models.setCursorPosition(0)

    self.log_mode = QComboBox()
    self.log_mode.addItems([
        "Standard (INFO)",
        "Signal only (scores)",
        "Verbose (--debug)",
        "Raw unfiltered (--debug)",
    ])
    self.log_mode.setCurrentIndex(0)
    self.log_mode.currentIndexChanged.connect(self._on_log_mode_changed)

    self.seed = QCheckBox("Seed from best (hill climb)")
    self.seed.setChecked(False)
    self.respawn = QCheckBox("Respawn excluded students")
    self.respawn.setChecked(True)
    self.insights = QCheckBox("Cross-run insights")
    self.insights.setChecked(True)
    self.deploy = QCheckBox("Promote winner to Arduino")
    self.deploy.setChecked(True)
    self.hil_check = QCheckBox("Auto-flash & HIL on win")
    self.hil_check.setChecked(True)

    self.run_btn = QPushButton("Run")
    self.run_btn.setStyleSheet(AppStyles.get_button_style())
    self.run_btn.clicked.connect(self.run_codegen)
    self.mock_btn = QPushButton("Mock Run")
    self.mock_btn.setStyleSheet(AppStyles.get_button_style())
    self.mock_btn.clicked.connect(lambda: self.run_codegen(mock=True))
    self.watch_btn = QPushButton("Auto-Watch")
    self.watch_btn.setToolTip("Watch for new gesture recordings and auto-run")
    self.watch_btn.setStyleSheet(AppStyles.get_button_style())
    self.watch_btn.clicked.connect(self.run_watch)
    self.stop_btn = QPushButton("Stop")
    self.stop_btn.setStyleSheet(AppStyles.get_button_style())
    self.stop_btn.clicked.connect(self.stop)
    self.stop_btn.setEnabled(False)

    box = QGroupBox("Codegen Configuration")
    box.setStyleSheet(AppStyles.get_group_box_style())
    lay = QVBoxLayout(box)

    grid = QGridLayout()
    for r, (label_text, widget) in enumerate([
        ("Students:", self.students_spin),
        ("Cycles:", self.cycles),
        ("CV Folds:", self.cv_folds_spin),
        ("Timeout:", self.timeout_spin),
    ]):
      lbl = QLabel(label_text)
      lbl.setStyleSheet(AppStyles.get_label_style())
      col = (r % 2) * 2
      row_idx = r // 2
      grid.addWidget(lbl, row_idx, col)
      grid.addWidget(widget, row_idx, col + 1)
    lay.addLayout(grid)

    lbl_m = QLabel("Student models (comma-sep):")
    lbl_m.setStyleSheet(AppStyles.get_label_style())
    lay.addWidget(lbl_m)
    lay.addWidget(self.models)

    log_row = QHBoxLayout()
    lbl_log = QLabel("Log mode:")
    lbl_log.setStyleSheet(AppStyles.get_label_style())
    log_row.addWidget(lbl_log)
    log_row.addWidget(self.log_mode, 1)
    lay.addLayout(log_row)

    lay.addWidget(self.seed)
    lay.addWidget(self.respawn)
    lay.addWidget(self.insights)
    lay.addWidget(self.deploy)
    lay.addWidget(self.hil_check)

    btn_row1 = QHBoxLayout()
    btn_row1.addWidget(self.run_btn)
    btn_row1.addWidget(self.mock_btn)
    lay.addLayout(btn_row1)

    btn_row2 = QHBoxLayout()
    btn_row2.addWidget(self.watch_btn)
    btn_row2.addWidget(self.stop_btn)
    lay.addLayout(btn_row2)
    return box

  def _device_box(self) -> QGroupBox:
    self.retrain_btn = QPushButton("Fit & Export Selected")
    self.retrain_btn.setToolTip(
        "Re-fit the selected Leaderboard solution (or top solution) on"
        " Python/data and export to Arduino/Codegen/model_data.h"
    )
    self.retrain_btn.setStyleSheet(AppStyles.get_button_style())
    self.retrain_btn.clicked.connect(self.retrain_selected)
    self.flash_btn = QPushButton("Flash Firmware")
    self.flash_btn.setStyleSheet(AppStyles.get_button_style())
    self.flash_btn.clicked.connect(self.flash)
    self.eval_btn = QPushButton("Evaluate on Device (HIL)")
    self.eval_btn.setStyleSheet(AppStyles.get_button_style())
    self.eval_btn.clicked.connect(self.device_eval)
    self.report_btn = QPushButton("View Latest Report")
    self.report_btn.setStyleSheet(AppStyles.get_button_style())
    self.report_btn.clicked.connect(self.open_report)
    note = QLabel(
        "Playground serial connection is automatically released before"
        " flashing or on-device evaluation."
    )
    note.setStyleSheet(AppStyles.get_label_style())
    note.setWordWrap(True)
    box = QGroupBox("Device & Reports")
    box.setStyleSheet(AppStyles.get_group_box_style())
    lay = QVBoxLayout(box)
    lay.addWidget(self.retrain_btn)
    lay.addWidget(self.flash_btn)
    lay.addWidget(self.eval_btn)
    lay.addWidget(self.report_btn)
    lay.addWidget(note)
    return box

  # -- state ------------------------------------------------------------------

  def refresh(self) -> None:
    try:
      from codegen_task import data_loader  # pylint: disable=import-outside-toplevel
      recs = data_loader.list_recordings(data_loader.default_data_dir())
      counts: dict[str, int] = {}
      for r in recs:
        counts[r.label] = counts.get(r.label, 0) + 1
      fp = data_loader.dataset_fingerprint(data_loader.default_data_dir())
      lines = [f"{len(recs)} recordings ({len(counts)} classes), fp {fp}"]
      pairs = [f"{k}: {v}" for k, v in sorted(counts.items())]
      for i in range(0, len(pairs), 3):
        lines.append("  " + " | ".join(pairs[i:i + 3]))
      thin = [k for k, v in counts.items() if v < 8]
      if thin:
        lines.append(f"Thin classes (<8 takes): {', '.join(sorted(thin))}")
      self.dataset_label.setText("\n".join(lines))
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.dataset_label.setText(f"Dataset unavailable: {e}")

    dep_path = (
        DEPLOYED_JSON
        if os.path.isfile(DEPLOYED_JSON)
        else DEPLOYED_FALLBACK_JSON
    )
    if os.path.isfile(dep_path):
      with open(dep_path, encoding="utf-8") as f:
        d = json.load(f)
      a = d.get("artifact", {})
      m = d.get("test_metrics", {})
      self.deployed_label.setText(
          f"Deployed: score {d.get('test_overall_score', 0):.3f}, "
          f"F1 {m.get('f1_macro', 0):.3f}, {a.get('model_kb', 0):.1f} KB, "
          f"{a.get('mflops', 0):.3f} MFLOPs, int8 {a.get('quantized')}, "
          f"dataset {d.get('dataset_fingerprint', '?')}"
      )
    else:
      self.deployed_label.setText("Deployed: none yet (fresh start)")
    self._load_leaderboard()
    self._refresh_reports_list()
    if self.process is None and not self._live_records:
      self._load_latest_run_metrics()
    self._update_bar_plot()

  def _load_leaderboard(self) -> None:
    self.table.clear()
    if not os.path.isfile(LEADERBOARD):
      self.table.setRowCount(0)
      return
    with open(LEADERBOARD, encoding="utf-8") as f:
      rows = [l.rstrip("\n").split("\t") for l in f if l.strip()]
    if not rows:
      return
    header, body = rows[0], rows[1:]
    col_idx = {h: i for i, h in enumerate(header)}
    preferred = [
        ("student", "Student"),
        ("elo", "Elo"),
        ("test_overall", "Score"),
        ("test_f1_macro", "F1"),
        ("test_wcr", "Worst Recall"),
        ("test_bacc", "Bal Acc"),
        ("model_kb", "KB"),
        ("mflops", "MFLOPs"),
        ("arena_kb", "Arena KB"),
        ("host_ms", "Host ms"),
        ("int8", "INT8"),
        ("cycles", "Cycles"),
        ("seeded", "Seeded"),
        ("elapsed_s", "Time (s)"),
        ("timestamp", "Timestamp"),
        ("solution_id", "Solution"),
        ("dataset", "Dataset"),
        ("tags", "Tags"),
        ("ops", "Ops"),
    ]
    cols = [(col_idx[k], label, k) for k, label in preferred if k in col_idx]
    self.table.setColumnCount(len(cols))
    self.table.setHorizontalHeaderLabels([label for _, label, _ in cols])
    self.table.setRowCount(len(body))
    left_align_keys = {"student", "tags", "ops", "solution_id"}
    for r, row in enumerate(reversed(body)):
      for c, (i, _, k) in enumerate(cols):
        val = row[i] if i < len(row) else ""
        item = QTableWidgetItem(val)
        if k in left_align_keys:
          item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        else:
          item.setTextAlignment(Qt.AlignCenter)
        item.setData(Qt.UserRole, row)
        self.table.setItem(r, c, item)
    self.table.resizeColumnsToContents()

  def _on_leaderboard_row_clicked(self, row_idx: int, _col_idx: int) -> None:
    item = self.table.item(row_idx, 0)
    if item is None:
      return
    row = item.data(Qt.UserRole) or []
    for text in row:
      if isinstance(text, str) and text.startswith("micro_gesture_"):
        cand = os.path.join(SOLUTIONS_DIR, f"{text}.md")
        if os.path.isfile(cand):
          self._show_report_path(cand)
          return
      if isinstance(text, str) and "/" in text:
        cand = os.path.join(EXP_DIR, "micro_gesture", text, "export", "REPORT.md")
        if os.path.isfile(cand):
          self._show_report_path(cand)
          return

  def _discover_reports(self) -> list[tuple[str, str]]:
    found: list[tuple[float, str, str]] = []
    for p in glob.glob(os.path.join(SOLUTIONS_DIR, "*.md")):
      found.append((os.path.getmtime(p), f"Solution: {os.path.basename(p)}", p))
    for p in glob.glob(os.path.join(EXP_DIR, "micro_gesture", "*", "*", "export", "REPORT.md")):
      rel = os.path.relpath(os.path.dirname(os.path.dirname(p)), EXP_DIR)
      found.append((os.path.getmtime(p), f"Run Report: {rel}", p))
    for p in glob.glob(os.path.join(EXP_DIR, "archive_pre_12class", "solutions", "*.md")):
      found.append((os.path.getmtime(p), f"Archived: {os.path.basename(p)}", p))
    prompt_md = os.path.join(TASK_DIR, "PROMPT.md")
    if os.path.isfile(prompt_md):
      found.append((0.0, "Task Prompt (PROMPT.md)", prompt_md))
    readme_md = os.path.join(TASK_DIR, "README.md")
    if os.path.isfile(readme_md):
      found.append((0.1, "Task & Gesture Guide (README.md)", readme_md))
    found.sort(key=lambda t: t[0], reverse=True)
    return [(label, path) for _, label, path in found]

  def _refresh_reports_list(self) -> None:
    prev_path = self._current_report_path()
    self._report_paths = self._discover_reports()
    self.report_combo.blockSignals(True)
    self.report_combo.clear()
    for label, path in self._report_paths:
      self.report_combo.addItem(label, path)
    idx = 0
    if prev_path:
      for i, (_, p) in enumerate(self._report_paths):
        if p == prev_path:
          idx = i
          break
    self.report_combo.setCurrentIndex(idx)
    self.report_combo.blockSignals(False)
    if self._report_paths:
      self._render_markdown_file(self._report_paths[idx][1])

  def _current_report_path(self) -> str | None:
    idx = self.report_combo.currentIndex()
    if 0 <= idx < len(self._report_paths):
      return self._report_paths[idx][1]
    return None

  def _on_main_tab_changed(self, idx: int) -> None:
    if idx == 0:
      self._poll_active_run()
    elif idx == 1 and not self._is_any_run_currently_active():
      cur = self._current_report_path()
      if cur and os.path.isfile(cur):
        self._render_markdown_file(cur)

  def _on_report_selected(self, idx: int) -> None:
    if 0 <= idx < len(self._report_paths):
      self._render_markdown_file(self._report_paths[idx][1])

  def _on_log_mode_changed(self, _idx: int) -> None:
    cur = self._current_report_path()
    if cur and os.path.isfile(cur):
      self._render_markdown_file(cur)

  def _show_report_path(self, path: str) -> None:
    self._refresh_reports_list()
    for i, (_, p) in enumerate(self._report_paths):
      if os.path.abspath(p) == os.path.abspath(path):
        self.report_combo.setCurrentIndex(i)
        break
    self.main_tabs.setCurrentIndex(1)
    self._render_markdown_file(path)

  def _resolve_run_dir_from_report(self, path: str, text: str) -> str | None:
    abs_path = os.path.abspath(path)
    parent = os.path.dirname(abs_path)
    if os.path.basename(parent) == "export":
      cand = os.path.dirname(parent)
      if os.path.isdir(cand):
        return cand
    m_ts = re.search(
        r"(?:Run\s+`|\*\*Run\*\*:\s*`)([0-9]{8}_[0-9]{6})`", text
    )
    if m_ts:
      cand = os.path.join(
          EXP_DIR, "micro_gesture", "micro_gesture", m_ts.group(1)
      )
      if os.path.isdir(cand):
        return cand
    m_rel = re.search(r"\*\*Run Path\*\*:\s*`?([^\s`\n]+)`?", text)
    if m_rel:
      cand = os.path.join(EXP_DIR, "micro_gesture", m_rel.group(1))
      if os.path.isdir(cand):
        return cand
    sol_id = os.path.splitext(os.path.basename(abs_path))[0]
    if os.path.isfile(LEADERBOARD):
      try:
        with open(LEADERBOARD, encoding="utf-8") as f:
          rows = [l.rstrip("\n").split("\t") for l in f if l.strip()]
        if rows:
          hdr = rows[0]
          s_idx = hdr.index("solution_id") if "solution_id" in hdr else -1
          r_idx = hdr.index("run_dir") if "run_dir" in hdr else -1
          if s_idx >= 0 and r_idx >= 0:
            for row in reversed(rows[1:]):
              if len(row) > max(s_idx, r_idx) and row[s_idx] == sol_id:
                cand = os.path.join(EXP_DIR, "micro_gesture", row[r_idx])
                if os.path.isdir(cand):
                  return cand
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    return None

  def _filter_run_log_lines(self, log_path: str) -> list[str]:
    mode_idx = self.log_mode.currentIndex() if hasattr(self, "log_mode") else 0
    kept: list[str] = []
    in_debug_block = False
    try:
      with open(log_path, encoding="utf-8", errors="ignore") as f:
        for raw in f:
          line = raw.rstrip("\r\n")
          m_lvl = _LOG_LEVEL_LINE_RE.match(line)
          if m_lvl:
            in_debug_block = m_lvl.group(1) == "DEBUG"
          elif re.match(r"^[IWE]\d{4}\s+", line):
            in_debug_block = False
          if mode_idx in (0, 1) and in_debug_block:
            continue
          if self._should_display_line(line):
            kept.append(line)
    except Exception:  # pylint: disable=broad-exception-caught
      return []
    max_lines = 1500
    if len(kept) > max_lines:
      omitted = len(kept) - max_lines
      kept = (
          kept[:300]
          + [f"... ({omitted} verbose lines omitted; see {log_path}) ..."]
          + kept[-1200:]
      )
    return kept

  def _append_run_log_to_report(self, run_dir: str) -> None:
    log_path = os.path.join(run_dir, "run.log")
    if not os.path.isfile(log_path):
      return
    lines = self._filter_run_log_lines(log_path)
    if not lines:
      return
    mode_label = (
        self.log_mode.currentText()
        if hasattr(self, "log_mode")
        else "Standard (INFO)"
    )
    rel_log = os.path.relpath(log_path, REPO_DIR)
    colored_lines = "\n".join(_colorize_log_line_html(l) for l in lines)
    log_html = (
        "<hr style='border: 1px solid #3B4252; margin-top: 18px;'/>"
        "<h2>Run Terminal Log</h2>"
        f"<p style='font-size: 11px; color: #94A3B8;'>"
        f"Source: <code>{html.escape(rel_log)}</code> &mdash; "
        f"{len(lines)} lines shown (Filter: <b>{html.escape(mode_label)}</b>)"
        "</p>"
        "<pre style='background-color: #1E222A; color: #D8DEE9;"
        " font-family: Menlo, Monaco, monospace; font-size: 10px;"
        " line-height: 1.38; padding: 10px; border-radius: 6px;"
        " border: 1px solid #2E3440; white-space: pre-wrap;'>"
        f"{colored_lines}</pre>"
    )
    cur = QTextCursor(self.report_browser.document())
    cur.movePosition(QTextCursor.End)
    cur.insertHtml(log_html)

  def _render_markdown_file(self, path: str) -> None:
    if not os.path.isfile(path):
      self.report_browser.setPlainText(f"File not found: {path}")
      return
    try:
      with open(path, encoding="utf-8") as f:
        text = f.read()
      base_dir = os.path.dirname(os.path.abspath(path))
      search_dirs = [base_dir, SOLUTIONS_DIR, TASK_DIR]
      self.report_browser.setSearchPaths(search_dirs)
      self.report_browser.document().setBaseUrl(
          QUrl.fromLocalFile(base_dir + "/")
      )
      if hasattr(self.report_browser, "setMarkdown"):
        self.report_browser.setMarkdown(text)
        self._scale_markdown_images(search_dirs, max_width=740)
        run_dir = self._resolve_run_dir_from_report(path, text)
        if run_dir:
          self._append_run_log_to_report(run_dir)
          self._pinned_report_run_dir = run_dir
          if (
              self.process is None
              and not self._is_any_run_currently_active()
              and self.main_tabs.currentIndex() == 1
          ):
            self._load_run_dir_metrics(run_dir)
      else:
        self.report_browser.setPlainText(text)
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.report_browser.setPlainText(f"Could not render {path}: {e}")

  def _scale_markdown_images(self, search_dirs: list[str],
                             max_width: int = 740) -> None:
    doc = self.report_browser.document()
    base_url = doc.baseUrl()
    dark_mode = bool(
        getattr(self, "report_dark_mode", None)
        and self.report_dark_mode.isChecked()
    )
    cur = QTextCursor(doc)
    block = doc.begin()
    while block.isValid():
      it = block.begin()
      while not it.atEnd():
        frag = it.fragment()
        fmt = frag.charFormat()
        if fmt.isImageFormat():
          ifmt = fmt.toImageFormat()
          name = ifmt.name()
          img_path = None
          for d in search_dirs:
            cand = os.path.join(d, name)
            if os.path.isfile(cand):
              img_path = cand
              break
          if not img_path and name.startswith("figs/"):
            bname = os.path.basename(name)
            matches = glob.glob(os.path.join(SOLUTIONS_DIR, "plots", f"*_{bname}"))
            if matches:
              img_path = max(matches, key=os.path.getmtime)
          if img_path:
            img = QImage(img_path)
            if not img.isNull():
              # Give multi-column trajectory / dashboard plots extra width
              is_wide_plot = (
                  any(k in name for k in (
                      "progress", "elo_ratings", "pareto", "curves",
                      "class_signatures", "macs_per_op", "stream",
                  ))
                  or (name.startswith("plots/micro_gesture_")
                      and name.count("_") <= 4)
              )
              target_max_w = max_width if is_wide_plot else min(620, max_width)
              w = min(target_max_w, img.width())
              h = int(round(img.height() * (w / max(1, img.width()))))
              scaled = img.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
              if dark_mode:
                scaled = _invert_plot_for_dark_mode(scaled)
              doc.addResource(QTextDocument.ImageResource, QUrl(name), scaled)
              doc.addResource(
                  QTextDocument.ImageResource, base_url.resolved(QUrl(name)), scaled
              )
              ifmt.setWidth(w)
              ifmt.setHeight(h)
              cur.setPosition(frag.position())
              cur.setPosition(
                  frag.position() + frag.length(), QTextCursor.KeepAnchor
              )
              cur.setCharFormat(ifmt)
        it += 1
      block = block.next()

  # -- live metrics graph -----------------------------------------------------

  def _clear_metrics_graph(self) -> None:
    self._live_records.clear()
    self._collab_cycles.clear()
    self._loaded_run_dir = None
    for curve in self._metric_curves.values():
      curve.setData([], [])
    self.student_plot.clear()

  def _short_student_name(self, st: str) -> str:
    m = re.match(r"student_(\d+)_(?:gemini_)?(.+)", st)
    if m:
      idx, model = m.group(1), m.group(2)
      short = (
          model.replace("3_1_pro_preview", "pro")
          .replace("3.1-pro-preview", "pro")
          .replace("3_5_flash_lite", "lite")
          .replace("3.5-flash-lite", "lite")
          .replace("3_5_flash", "flash")
          .replace("3.5-flash", "flash")
          .replace("4_argon", "argon")
          .replace("4-argon", "argon")
      )
      return f"S{idx}:{short}"
    return st

  def _short_tag_label(self, tag: str) -> str:
    t = tag.strip().lower()
    mapping = {
        "in-graph-norm": "in-norm",
        "depthwise-conv": "dw-conv",
        "ds-conv2d": "ds-conv",
        "hard-negatives": "hard-neg",
        "cosine-decay": "cos-lr",
        "dual-global-pooling": "dual-pool",
        "global-pooling-concat": "pool-cat",
        "peak-aligned-blending": "peak-blend",
        "data-augmentation": "data-aug",
        "physics-augmentation": "phys-aug",
        "2-segment-pooling": "2seg-pool",
        "multi-segment-pooling": "mseg-pool",
        "hybrid-pooling": "hyb-pool",
        "temporal-downsampling": "temp-ds",
        "label-smoothing": "lbl-smth",
        "kinematic-features": "kinematic",
        "dual-branch-ensemble": "2-branch",
        "multi-scale-conv2d": "ms-conv",
        "strided-conv": "stride-cv",
    }
    return mapping.get(t, t[:10])

  def _short_op_label(self, op: str) -> str:
    o = op.strip().upper()
    mapping = {
        "FULLY_CONNECTED": "FC",
        "CONCATENATION": "CONCAT",
        "DEPTHWISE_CONV_2D": "DW_CONV",
        "AVERAGE_POOL_2D": "AVG_POOL",
        "MAX_POOL_2D": "MAX_POOL",
        "STRIDED_SLICE": "SLICE",
        "REDUCE_MAX": "RED_MAX",
        "DEQUANTIZE": "DEQUANT",
        "QUANTIZE": "QUANT",
    }
    return mapping.get(o, o[:9])

  def _plot_leaderboard_histogram(self, mode_key: str) -> None:
    if not os.path.isfile(LEADERBOARD):
      return
    try:
      df = pd.read_csv(LEADERBOARD, sep="\t")
    except Exception:  # pylint: disable=broad-exception-caught
      return
    if df.empty:
      return
    col = "tags" if mode_key == "top_tags" else "ops"
    sep = "," if mode_key == "top_tags" else "|"
    if col not in df:
      return
    counts: dict[str, int] = {}
    score_sums: dict[str, float] = {}
    # Trivial boilerplate ops present in almost every classifier graph
    skip_ops = {"RESHAPE", "SOFTMAX"}
    for _, row in df.iterrows():
      raw = str(row.get(col, "") or "")
      if not raw or raw.lower() == "nan":
        continue
      sc = float(row.get("test_overall", 0.0) or 0.0)
      tokens = {
          tok.strip()
          for tok in raw.split(sep)
          if tok.strip() and tok.strip().lower() != "mock"
      }
      for tok in tokens:
        if mode_key == "top_ops" and tok.upper() in skip_ops:
          continue
        counts[tok] = counts.get(tok, 0) + 1
        score_sums[tok] = score_sums.get(tok, 0.0) + sc
    if not counts:
      return
    # Sort descending by frequency, then by mean test score on ties
    ranked = sorted(
        counts.keys(),
        key=lambda k: (counts[k], score_sums[k] / max(1, counts[k])),
        reverse=True,
    )[:8]
    x_lbl = "Technique Tag" if mode_key == "top_tags" else "TFLite Operator"
    self.student_plot.getAxis("bottom").setLabel(
        x_lbl, color=AppStyles.GRAPH_LABEL_COLOR, **{"font-size": "9pt"}
    )
    x_ticks = [
        (
            i,
            self._short_tag_label(k)
            if mode_key == "top_tags"
            else self._short_op_label(k),
        )
        for i, k in enumerate(ranked)
    ]
    self.student_plot.getAxis("bottom").setTicks([x_ticks])
    self.student_plot.setXRange(-0.6, len(ranked) - 0.4)
    for i, k in enumerate(ranked):
      color = _STUDENT_COLORS[i % len(_STUDENT_COLORS)]
      val = float(counts[k])
      bar = pg.BarGraphItem(
          x=[i], height=[val], width=0.56,
          brush=pg.mkBrush(color),
          pen=pg.mkPen("#FFFFFF", width=1),
      )
      self.student_plot.addItem(bar)
    max_v = float(max(counts[k] for k in ranked))
    self.student_plot.setYRange(0.0, max(2.0, max_v * 1.18))

  def _update_metrics_graph(self) -> None:
    for key, chk in self._metric_checks.items():
      if key in self._metric_curves:
        self._metric_curves[key].setVisible(chk.isChecked())

    if self._live_records:
      cycles = sorted({c for c, _ in self._live_records.keys()})
      max_c = max(cycles) if cycles else 0
      ticks = [(c, str(c)) for c in range(max_c + 1)]
      self.metrics_plot.getAxis("bottom").setTicks([ticks])
      self.metrics_plot.setXRange(-0.25, max(0.5, max_c + 0.25))

      # 1. Top Plot: Running-best bounded metrics across cycles (auto-zoomed Y)
      vis_y: list[float] = []
      for key, _label, _color, _def_on, low_better in _BOUNDED_METRICS:
        curve = self._metric_curves.get(key)
        if curve is None:
          continue
        xs, ys = [], []
        running: float | None = None
        for c in cycles:
          vals = [
              float(rec[key])
              for (rc, _st), rec in self._live_records.items()
              if rc == c and key in rec and not np.isnan(float(rec[key]))
          ]
          if vals:
            cyc_best = min(vals) if low_better else max(vals)
            if running is None:
              running = cyc_best
            else:
              running = (
                  min(running, cyc_best) if low_better else max(running, cyc_best)
              )
          if running is not None:
            xs.append(c)
            ys.append(running)
        curve.setData(xs, ys)
        if self._metric_checks[key].isChecked() and ys:
          vis_y.extend(ys)

      if vis_y:
        y_min, y_max = min(vis_y), max(vis_y)
        span = max(0.04, y_max - y_min)
        pad = 0.14 * span
        lo = max(-0.02, y_min - pad)
        hi = min(1.03, y_max + pad)
        if hi - lo < 0.05:
          mid = 0.5 * (lo + hi)
          lo, hi = max(-0.02, mid - 0.03), min(1.03, mid + 0.03)
        self.metrics_plot.setYRange(lo, hi)

    self._update_bar_plot()

  def _update_bar_plot(self) -> None:
    # 2. Bottom Plot: Per-Student Bar Summary or Leaderboard Histograms
    self.student_plot.clear()
    sel_idx = self.student_metric_combo.currentIndex()
    if not (0 <= sel_idx < len(_STUDENT_BAR_MODES)):
      sel_idx = 0
    mode_key, _mode_label, y_lbl = _STUDENT_BAR_MODES[sel_idx]
    self.student_plot.getAxis("left").setLabel(
        y_lbl,
        color=AppStyles.GRAPH_LABEL_COLOR,
        **{"font-size": "9pt"},
    )

    if mode_key in ("top_tags", "top_ops"):
      self._plot_leaderboard_histogram(mode_key)
      return

    self.student_plot.getAxis("bottom").setLabel(
        "Student", color=AppStyles.GRAPH_LABEL_COLOR, **{"font-size": "9pt"}
    )
    cycles = sorted({c for c, _ in self._live_records.keys()})
    students = sorted({st for _c, st in self._live_records.keys()})
    if not students:
      return

    # Compute per-cycle ranks across students (1 = best overall_score in cycle)
    cycle_ranks: dict[str, list[float]] = {st: [] for st in students}
    for c in cycles:
      c_scores = [
          (float(self._live_records[(c, st)]["overall_score"]), st)
          for st in students
          if (c, st) in self._live_records
          and "overall_score" in self._live_records[(c, st)]
          and not np.isnan(float(self._live_records[(c, st)]["overall_score"]))
      ]
      c_scores.sort(key=lambda t: t[0], reverse=True)
      for rank_idx, (_sc, st) in enumerate(c_scores, start=1):
        cycle_ranks[st].append(float(rank_idx))

    x_ticks = [(i, self._short_student_name(st)) for i, st in enumerate(students)]
    self.student_plot.getAxis("bottom").setTicks([x_ticks])
    self.student_plot.setXRange(-0.6, len(students) - 0.4)

    primary_vals: list[float] = []
    secondary_vals: list[float] = []
    has_secondary = mode_key in (
        "score_best_mean", "f1_best_mean", "wcr_best_mean", "cycles_collab",
    )

    for st in students:
      recs = [
          self._live_records[(c, st)]
          for c in cycles
          if (c, st) in self._live_records
      ]
      # Identify this student's best cycle by overall_score
      best_rec = max(
          recs, key=lambda r: float(r.get("overall_score", -1.0)), default={}
      )
      if mode_key == "score_best_mean":
        vals = [float(r["overall_score"]) for r in recs if "overall_score" in r]
        primary_vals.append(max(vals) if vals else 0.0)
        secondary_vals.append(float(np.mean(vals)) if vals else 0.0)
      elif mode_key == "f1_best_mean":
        vals = [float(r["f1_macro"]) for r in recs if "f1_macro" in r]
        primary_vals.append(max(vals) if vals else 0.0)
        secondary_vals.append(float(np.mean(vals)) if vals else 0.0)
      elif mode_key == "wcr_best_mean":
        vals = [
            float(r["worst_class_recall"])
            for r in recs
            if "worst_class_recall" in r
        ]
        primary_vals.append(max(vals) if vals else 0.0)
        secondary_vals.append(float(np.mean(vals)) if vals else 0.0)
      elif mode_key == "avg_rank":
        rks = cycle_ranks.get(st, [])
        primary_vals.append(float(np.mean(rks)) if rks else 0.0)
      elif mode_key == "code_loc":
        loc_best = float(best_rec.get("loc", 0.0))
        if loc_best <= 0:
          locs = [float(r["loc"]) for r in recs if r.get("loc", 0) > 0]
          loc_best = float(np.mean(locs)) if locs else 0.0
        primary_vals.append(loc_best)
      elif mode_key == "model_kb":
        primary_vals.append(float(best_rec.get("model_kb", 0.0)))
      elif mode_key == "mflops":
        primary_vals.append(float(best_rec.get("mflops", 0.0)))
      elif mode_key == "cycles_collab":
        n_scored = float(len(recs))
        n_col = float(sum(1 for r in recs if r.get("collab", False)))
        primary_vals.append(n_scored)
        secondary_vals.append(n_col)

    for i, st in enumerate(students):
      color = _STUDENT_COLORS[i % len(_STUDENT_COLORS)]
      p_val = primary_vals[i]
      if has_secondary:
        s_val = secondary_vals[i]
        # Solid bar = Best (or Total Cycles); hatched/dimmer bar = Mean (or Collabs)
        bar1 = pg.BarGraphItem(
            x=[i - 0.17], height=[p_val], width=0.30,
            brush=pg.mkBrush(color),
            pen=pg.mkPen("#FFFFFF", width=1),
        )
        dim_q = QColor(color)
        dim_q.setAlpha(135)
        bar2 = pg.BarGraphItem(
            x=[i + 0.17], height=[s_val], width=0.30,
            brush=pg.mkBrush(dim_q),
            pen=pg.mkPen(color, width=1.2, style=Qt.DashLine),
        )
        self.student_plot.addItem(bar1)
        self.student_plot.addItem(bar2)
      else:
        bar = pg.BarGraphItem(
            x=[i], height=[p_val], width=0.52,
            brush=pg.mkBrush(color),
            pen=pg.mkPen("#FFFFFF", width=1),
        )
        self.student_plot.addItem(bar)

    all_bar_vals = primary_vals + (secondary_vals if has_secondary else [])
    if all_bar_vals:
      b_min, b_max = min(all_bar_vals), max(all_bar_vals)
      if mode_key in ("score_best_mean", "f1_best_mean", "wcr_best_mean"):
        # Auto-zoom around the student range so small differences between
        # students stand out clearly instead of compressing into [0, 1].
        span = max(0.05, b_max - b_min)
        lo = max(0.0, b_min - 0.25 * span)
        hi = min(1.03, b_max + 0.18 * span)
        self.student_plot.setYRange(lo, hi)
      else:
        self.student_plot.setYRange(0.0, max(1.0, b_max * 1.16))

  def _find_newest_run_dir(self) -> tuple[str | None, float]:
    """Returns (newest_run_dir, latest_activity_mtime) across all runs."""
    candidates = glob.glob(
        os.path.join(EXP_DIR, "micro_gesture", "*", "*")
    )
    best_dir: str | None = None
    best_mtime = 0.0
    for d in candidates:
      if not os.path.isdir(d):
        continue
      csvs = glob.glob(os.path.join(d, "students", "*.csv"))
      log_p = os.path.join(d, "run.log")
      mtimes = [os.path.getmtime(p) for p in csvs if os.path.isfile(p)]
      if os.path.isfile(log_p):
        mtimes.append(os.path.getmtime(log_p))
      if not mtimes:
        continue
      mt = max(mtimes)
      if mt > best_mtime:
        best_mtime = mt
        best_dir = d
    return best_dir, best_mtime

  def _is_any_run_currently_active(self) -> bool:
    """True if a GUI child process is running or a run directory updated <90s ago."""
    if self.process is not None:
      return True
    import time  # pylint: disable=import-outside-toplevel
    _d, mt = self._find_newest_run_dir()
    return (time.time() - mt) < 90.0 if mt > 0 else False

  def _poll_active_run(self) -> None:
    if self.active_run_dir and os.path.isdir(self.active_run_dir):
      self._load_run_dir_metrics(self.active_run_dir)
      return
    newest_dir, mt = self._find_newest_run_dir()
    if not newest_dir:
      return
    if self.process is not None and self._run_launch_time > 0.0:
      # While a newly started classroom process is still importing TensorFlow
      # (before creating its new run_dir), do not load a prior run directory.
      if mt < self._run_launch_time:
        return
    import time  # pylint: disable=import-outside-toplevel
    is_live = (time.time() - mt) < 90.0
    # When a run is live (or the user is on the Live Run Log tab), always track
    # the newest run directory. Only stay on a pinned historical report when no
    # live run is active and the user is browsing the Solution tab.
    if is_live or self.main_tabs.currentIndex() == 0:
      self._load_run_dir_metrics(newest_dir)
    elif self._pinned_report_run_dir and os.path.isdir(self._pinned_report_run_dir):
      self._load_run_dir_metrics(self._pinned_report_run_dir)
    else:
      self._load_run_dir_metrics(newest_dir)

  def _load_latest_run_metrics(self) -> None:
    newest_dir, _mt = self._find_newest_run_dir()
    if newest_dir:
      self._load_run_dir_metrics(newest_dir)

  def _load_run_dir_metrics(self, run_dir: str) -> None:
    norm_dir = os.path.abspath(run_dir)
    if self._loaded_run_dir and self._loaded_run_dir != norm_dir:
      self._clear_metrics_graph()
    self._loaded_run_dir = norm_dir
    if hasattr(self, "_metrics_box"):
      run_tag = os.path.basename(norm_dir)
      self._metrics_box.setTitle(f"Real-Time Cycle Trajectory (CV) — {run_tag}")
    csvs = glob.glob(os.path.join(norm_dir, "students", "*.csv"))
    if not csvs:
      return
    updated = False
    for path in csvs:
      try:
        df = pd.read_csv(path)
      except Exception:  # pylint: disable=broad-exception-caught
        continue
      if "cycle" not in df or "overall_score" not in df:
        continue
      st_name = os.path.splitext(os.path.basename(path))[0]
      for _, row in df.iterrows():
        try:
          c = int(row["cycle"])
          ov = float(row["overall_score"])
        except (ValueError, TypeError):
          continue
        if np.isnan(ov):
          continue
        rec = dict(self._live_records.get((c, st_name), {}))
        rec["overall_score"] = ov
        for col in df.columns:
          if col.startswith("metrics_"):
            m_name = col[len("metrics_"):]
            try:
              rec[m_name] = float(row[col])
            except (ValueError, TypeError):
              pass
        code_str = str(row.get("code", "") or "")
        if code_str and code_str.lower() != "nan":
          rec["loc"] = float(
              sum(1 for ln in code_str.splitlines() if ln.strip())
          )
        collab_val = str(row.get("collaborating_agent", "")).strip()
        mut_val = str(row.get("mutation_strategy", "")).strip()
        if (collab_val and collab_val.lower() != "nan") or ("PEER" in mut_val.upper()):
          rec["collab"] = True
          self._collab_cycles.add((c, st_name))
        self._live_records[(c, st_name)] = rec
        updated = True
    if updated:
      self._update_metrics_graph()

  # -- processes --------------------------------------------------------------

  def _release_serial_if_busy(self) -> None:
    h = self.serial_handler
    if h is not None and (getattr(h, "running", False) or h.isRunning()):
      self._append(
          "Releasing Playground serial connection so device task can access"
          " the board..."
      )
      h.disconnect()

  def _start(self, program: str, args: list[str], cwd: str,
             raw_logs: bool = False) -> None:
    if self.process is not None:
      self._append("A process is already running; stop it first.")
      return
    self.main_tabs.setCurrentIndex(0)
    self.log.clear()
    self._append(f"$ {program} {' '.join(args)}")
    proc = QProcess(self)
    env = QProcessEnvironment.systemEnvironment()
    env.insert("TF_CPP_MIN_LOG_LEVEL", "0" if raw_logs else "3")
    env.insert("PYTHONUNBUFFERED", "1")
    proc.setProcessEnvironment(env)
    proc.setWorkingDirectory(cwd)
    proc.setProcessChannelMode(QProcess.MergedChannels)
    proc.readyReadStandardOutput.connect(self._on_output)
    proc.finished.connect(self._on_finished)
    proc.start(program, args)
    self.process = proc
    self._set_running(True)
    self._poll_timer.start()

  def run_codegen(self, mock: bool = False) -> None:
    if self.deploy.isChecked() and self.hil_check.isChecked() and not mock:
      self._release_serial_if_busy()
    import time  # pylint: disable=import-outside-toplevel
    self._clear_metrics_graph()
    self.active_run_dir = None
    self._run_launch_time = time.time()
    if hasattr(self, "_metrics_box"):
      self._metrics_box.setTitle("Real-Time Cycle Trajectory (CV) — starting...")

    mode_idx = self.log_mode.currentIndex()
    debug_on = mode_idx in (2, 3)
    raw_on = mode_idx == 3

    args = [
        "-m", "codegen_task.run_micro_gesture",
        f"--num_students={self.students_spin.value()}",
        f"--num_cycles={1 if mock else self.cycles.value()}",
        f"--cv_folds={self.cv_folds_spin.value()}",
        f"--timeout_s={self.timeout_spin.value()}",
        f"--hil={'auto' if self.hil_check.isChecked() else 'off'}",
    ]
    if mock:
      args.append("--mock_students")
    else:
      args.append(f"--model_names={self.models.text().strip()}")
    if self.deploy.isChecked():
      args.append("--deploy")
    if not self.seed.isChecked():
      args.append("--no_seed")
    if not self.respawn.isChecked():
      args.append("--norespawn")
    if not self.insights.isChecked():
      args.append("--noinsights")
    if debug_on:
      args.append("--debug")
    self._start(sys.executable, args, PY_DIR, raw_logs=raw_on)

  run_classroom = run_codegen

  def run_watch(self) -> None:
    args = [
        "-m", "codegen_task.run_micro_gesture", "--watch",
        f"--num_students={self.students_spin.value()}",
        f"--num_cycles={self.cycles.value()}",
        f"--cv_folds={self.cv_folds_spin.value()}",
        f"--timeout_s={self.timeout_spin.value()}",
        f"--hil={'auto' if self.hil_check.isChecked() else 'off'}",
        f"--model_names={self.models.text().strip()}",
    ]
    self._start(sys.executable, args, PY_DIR)

  def _selected_solution_id(self) -> str:
    row_idx = self.table.currentRow()
    if row_idx >= 0:
      item = self.table.item(row_idx, 0)
      if item is not None:
        row = item.data(Qt.UserRole) or []
        for text in row:
          if isinstance(text, str) and text.startswith("micro_gesture_"):
            return text
    return "best"

  def retrain_selected(self) -> None:
    sol_id = self._selected_solution_id()
    self._start(
        sys.executable,
        ["-m", "codegen_task.run_micro_gesture", f"--retrain_solution={sol_id}"],
        PY_DIR,
    )

  def flash(self) -> None:
    self._release_serial_if_busy()
    self._start("make", ["flash-codegen"], REPO_DIR)

  def device_eval(self) -> None:
    self._release_serial_if_busy()
    self._start(
        sys.executable,
        ["-m", "codegen_task.deploy", "--json",
         os.path.join(EXP_DIR, "device_eval", "device_eval_latest.json")],
        PY_DIR,
    )

  def open_report(self) -> None:
    self._refresh_reports_list()
    if not self._report_paths:
      self._append("No report found yet; run codegen first.")
      return
    self.report_combo.setCurrentIndex(0)
    self.main_tabs.setCurrentIndex(1)
    self._render_markdown_file(self._report_paths[0][1])

  def open_report_external(self) -> None:
    path = self._current_report_path()
    if not path or not os.path.isfile(path):
      self._append("No report selected to open.")
      return
    QProcess.startDetached("open", [path])

  def stop(self) -> None:
    if self.process is not None:
      self._append("Stopping...")
      self.process.terminate()
      if not self.process.waitForFinished(3000):
        self.process.kill()

  def _should_display_line(self, line: str) -> bool:
    s = line.strip()
    if not s or s in ("(", ")"):
      return False
    mode_idx = self.log_mode.currentIndex()
    if mode_idx == 3:  # Raw unfiltered
      return True
    if any(n in line for n in _NOISE):
      return False
    if mode_idx == 1:  # Signal only
      return any(p in line for p in _SIGNAL_PATTERNS)
    return True

  def _on_output(self) -> None:
    if self.process is None:
      return
    data = bytes(self.process.readAllStandardOutput()).decode(errors="ignore")
    for line in data.splitlines():
      m_dir = _RUN_DIR_RE.search(line)
      if m_dir:
        self.active_run_dir = m_dir.group(1)
      m_peer = _MUTATE_PEER_RE.search(line)
      if m_peer:
        st_p = m_peer.group(1)
        cyc_p = int(m_peer.group(3))
        self._collab_cycles.add((cyc_p, st_p))
      m_cyc = _CYCLE_METRIC_RE.search(line)
      if m_cyc:
        c = int(m_cyc.group(1))
        st = m_cyc.group(2)
        rec = dict(self._live_records.get((c, st), {}))
        rec["overall_score"] = float(m_cyc.group(3))
        rec["f1_macro"] = float(m_cyc.group(4))
        rec["worst_class_recall"] = float(m_cyc.group(5))
        if m_cyc.group(6) is not None:
          rec["null_fpr"] = float(m_cyc.group(6))
          rec["model_kb"] = float(m_cyc.group(7))
          rec["mflops"] = float(m_cyc.group(8))
          rec["arena_kb"] = float(m_cyc.group(9))
        if (c, st) in self._collab_cycles:
          rec["collab"] = True
        self._live_records[(c, st)] = rec
        self._update_metrics_graph()
      else:
        m_lib = _LIB_SCORE_LINE_RE.search(line)
        if m_lib:
          st = m_lib.group(1)
          ov = float(m_lib.group(2))
          metrics_str = m_lib.group(3)
          c = int(m_lib.group(4))
          rec = dict(self._live_records.get((c, st), {}))
          rec["overall_score"] = ov
          for k, v in _KV_FLOAT_RE.findall(metrics_str):
            try:
              rec[k] = float(v)
            except ValueError:
              pass
          if (c, st) in self._collab_cycles:
            rec["collab"] = True
          self._live_records[(c, st)] = rec
          self._update_metrics_graph()
      if self._should_display_line(line):
        self._append(line)

  def _on_finished(self, code: int, _status) -> None:
    self._poll_active_run()
    self._append(f"[exit {code}]")
    self.process = None
    self.active_run_dir = None
    self._run_launch_time = 0.0
    self._set_running(False)
    self.refresh()

  def _set_running(self, running: bool) -> None:
    for b in (self.run_btn, self.mock_btn, self.watch_btn, self.retrain_btn,
              self.flash_btn, self.eval_btn):
      b.setEnabled(not running)
    self.stop_btn.setEnabled(running)

  def _append(self, text: str) -> None:
    self.log.appendPlainText(text)
    self.log.verticalScrollBar().setValue(
        self.log.verticalScrollBar().maximum())


ClassroomWidget = CodegenWidget


def _natural_label_key(label: str) -> tuple[str, str, int]:
  stem = os.path.splitext(os.path.basename(label))[0]
  m = re.match(r"^(.*?)-(\d+)$", stem)
  if m:
    return (m.group(1).lower(), m.group(1), int(m.group(2)))
  return (stem.lower(), stem, -1)


def _upgrade_playground_widgets(window) -> None:
  """Applies 100 Hz recording, natural sort, and safe serial disconnect in-place."""
  rtg = getattr(window, "real_time_graph", None)
  sh = getattr(rtg, "serial_handler", None) if rtg else None
  sdw = getattr(window, "sample_data_widget", None)
  cdw = getattr(window, "collected_data_widget", None)
  rdw = getattr(window, "record_data_widget", None)

  # 1. RealTimeGraph: 100 Hz 8-second window (800 samples)
  if rtg is not None:
    rtg.sample_rate = 100
    rtg.time_data = np.linspace(
        0, rtg.window_size, rtg.window_size * rtg.sample_rate
    )
    rtg.x_data = np.zeros(rtg.window_size * rtg.sample_rate)
    rtg.y_data = np.zeros(rtg.window_size * rtg.sample_rate)
    rtg.z_data = np.zeros(rtg.window_size * rtg.sample_rate)

  # 2. ArduinoSerialHandler: short read timeout + thread-safe disconnect
  if sh is not None:
    orig_connect = sh.connect

    def _safe_connect() -> bool:
      ok = orig_connect()
      if ok and getattr(sh, "serial_port", None) is not None:
        sh.serial_port.timeout = 0.15
      return ok

    def _safe_disconnect() -> None:
      was_running = sh.running
      sh.running = False
      if sh.isRunning():
        sh.wait(600)
      port = getattr(sh, "serial_port", None)
      sh.serial_port = None
      if port is not None:
        try:
          if port.is_open:
            port.close()
        except Exception:  # pylint: disable=broad-exception-caught
          pass
      if was_running:
        sh.connection_status.emit(False, "Disconnected")

    sh.connect = _safe_connect
    sh.disconnect = _safe_disconnect

  # 3. SampleDataWidget: 100 Hz time axis (100 samples = 1.0 s)
  if sdw is not None:
    orig_display = sdw.display_sample

    def _display_100hz(data, label):
      orig_display(data, label)
      if data is not None and len(data) > 0:
        t = np.linspace(0, len(data) / 100.0, len(data))
        sdw.x_curve.setData(t, data[:, 0])
        sdw.y_curve.setData(t, data[:, 1])
        sdw.z_curve.setData(t, data[:, 2])

    sdw.display_sample = _display_100hz

  # 4. CollectedDataWidget: natural sort order (Null-2 before Null-10)
  if cdw is not None:
    labels = sorted(cdw.samples.keys(), key=_natural_label_key)
    cdw.sample_list.clear()
    for lbl in labels:
      cdw.sample_list.addItem(lbl)
    if labels:
      cdw.sample_list.setCurrentRow(0)
      cdw.on_sample_selected(0)

    orig_add_sample = cdw.add_sample

    def _sorted_add_sample(data, label):
      orig_add_sample(data, label)
      items = [
          cdw.sample_list.item(i).text()
          for i in range(cdw.sample_list.count())
      ]
      sorted_items = sorted(set(items), key=_natural_label_key)
      cdw.sample_list.clear()
      for lbl in sorted_items:
        cdw.sample_list.addItem(lbl)
      if label in sorted_items:
        cdw.sample_list.setCurrentRow(sorted_items.index(label))

    cdw.add_sample = _sorted_add_sample

  # 5. RecordDataWidget: non-overwriting file counter + 100 Hz direct capture
  if rdw is not None:
    data_dir = os.path.join(PY_DIR, "data")

    def _sync_counter_from_disk() -> None:
      if not os.path.isdir(data_dir):
        return
      for fname in os.listdir(data_dir):
        if not fname.endswith(".txt"):
          continue
        m = re.match(r"^(.*?)-(\d+)\.txt$", fname)
        if m:
          cls_name, idx = m.group(1), int(m.group(2))
          rdw.counter[cls_name] = max(rdw.counter.get(cls_name, 0), idx)

    _sync_counter_from_disk()
    orig_process = rdw.process_recorded_data

    def _safe_process_recorded_data():
      _sync_counter_from_disk()
      orig_process()

    rdw.process_recorded_data = _safe_process_recorded_data

    if rtg is not None and sh is not None:
      def _capture_serial_sample(x: float, y: float, z: float) -> None:
        rdw.recording_buffer.append(np.array([x, y, z], dtype=float))

      orig_toggle = rdw.toggle_recording
      orig_stop = rdw.stop_recording

      def _toggle_100hz() -> None:
        was_rec = rdw.is_recording
        orig_toggle()
        if not was_rec and rdw.is_recording:
          use_hw = (
              rtg.data_source == rtg.USE_ARDUINO_DATA
              and getattr(sh, "running", False)
          )
          rdw._serial_direct = use_hw  # pylint: disable=protected-access
          if use_hw:
            try:
              rtg.data_updated.disconnect(rdw.capture_data)
            except TypeError:
              pass
            sh.data_received.connect(_capture_serial_sample)

      def _stop_100hz() -> None:
        if getattr(rdw, "_serial_direct", False):
          rdw._serial_direct = False  # pylint: disable=protected-access
          try:
            sh.data_received.disconnect(_capture_serial_sample)
          except TypeError:
            pass
          rtg.data_updated.connect(rdw.capture_data)
        orig_stop()

      rdw.toggle_recording = _toggle_100hz
      rdw.stop_recording = _stop_100hz

  # 6. TrainingWidget: add "Model Codegen (Selected Leaderboard Solution)" option
  tw = getattr(window, "training_widget", None)
  if tw is not None and not hasattr(tw, "model_codegen_checkbox"):
    tw.model_codegen_checkbox = QCheckBox(
        "Model Codegen (Selected Leaderboard Solution)"
    )
    tw_lay = tw.layout()
    if tw_lay is not None:
      btn_idx = tw_lay.indexOf(tw.train_button)
      if btn_idx >= 0:
        tw_lay.insertWidget(btn_idx, tw.model_codegen_checkbox)
      else:
        tw_lay.addWidget(tw.model_codegen_checkbox)

    all_boxes = (
        tw.model_nn_checkbox,
        tw.model_1dcnn_checkbox,
        tw.model_codegen_checkbox,
    )

    def _update_three_checkboxes(checkbox):
      if checkbox.isChecked():
        for b in all_boxes:
          if b != checkbox:
            b.setChecked(False)
        checkbox.setChecked(True)
      else:
        checkbox.setChecked(True)

    tw.update_checkboxes = _update_three_checkboxes
    tw.model_codegen_checkbox.clicked.connect(
        lambda: tw.update_checkboxes(tw.model_codegen_checkbox)
    )

    orig_train_model = tw.train_model

    def _train_model_with_codegen():
      if not tw.model_codegen_checkbox.isChecked():
        orig_train_model()
        return
      cw = getattr(window, "codegen_widget", None)
      sol_id = cw._selected_solution_id() if cw is not None else "best"
      tw.train_button.setEnabled(False)
      tw.status_label.setText(f"Status: Fitting Codegen solution ({sol_id})...")
      proc = QProcess(tw)
      env = QProcessEnvironment.systemEnvironment()
      env.insert("TF_CPP_MIN_LOG_LEVEL", "3")
      env.insert("PYTHONUNBUFFERED", "1")
      proc.setProcessEnvironment(env)
      proc.setWorkingDirectory(PY_DIR)
      proc.setProcessChannelMode(QProcess.MergedChannels)

      def _on_codegen_fit_done(code_exit: int, _status) -> None:
        tw.train_button.setEnabled(True)
        if cw is not None:
          cw.refresh()
        if code_exit == 0 and os.path.isfile(DEPLOYED_JSON):
          with open(DEPLOYED_JSON, encoding="utf-8") as f:
            d = json.load(f)
          tm = d.get("test_metrics", {})
          f1 = float(tm.get("f1_macro", 0.0)) * 100.0
          bacc = float(tm.get("balanced_accuracy", f1 / 100.0)) * 100.0
          sid = d.get("solution_id", sol_id)
          tw.status_label.setText(
              f"Status: Exported {sid} (Test F1: {f1:.1f}%)"
          )
          cm_list = d.get("confusion_matrix")
          cls_names = d.get("class_names", [])
          if cm_list and cls_names:
            try:
              from training_widget import ConfusionMatrixDialog  # pylint: disable=import-outside-toplevel
              dlg = ConfusionMatrixDialog(
                  np.array(cm_list, dtype=int), cls_names, bacc, tw
              )
              dlg.exec_()
            except Exception:  # pylint: disable=broad-exception-caught
              pass
        else:
          tw.status_label.setText(f"Status: Codegen fit exited ({code_exit})")

      proc.finished.connect(_on_codegen_fit_done)
      proc.start(
          sys.executable,
          ["-m", "codegen_task.run_micro_gesture", f"--retrain_solution={sol_id}"],
      )
      tw._codegen_proc = proc  # pylint: disable=protected-access

    tw.train_model = _train_model_with_codegen


def attach_codegen_tab(window) -> CodegenWidget:
  """Attaches the Codegen tab to MainWindow with minimal footprint on main_window.py."""
  _upgrade_playground_widgets(window)
  central_widget = window.takeCentralWidget()
  tabs = QTabWidget(window)
  tabs.addTab(central_widget, "Playground")
  serial_handler = getattr(
      getattr(window, "real_time_graph", None), "serial_handler", None
  )
  codegen = CodegenWidget(serial_handler)
  tabs.addTab(codegen, "Codegen")
  window.tabs = tabs
  window.codegen_widget = codegen
  window.classroom_widget = codegen
  window.setCentralWidget(tabs)
  return codegen


attach_classroom_tab = attach_codegen_tab

