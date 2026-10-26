"""Live Inference tab: real-time MCU gesture classification, top-k probability bars, IMU waveform, and serial monitor."""

from __future__ import annotations

from collections import deque
import datetime
import json
import os
import re
from typing import Any

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
import numpy as np
import pyqtgraph as pg

from app_styles import AppStyles

PY_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PY_DIR)
DEPLOYED_JSON = os.path.join(REPO_DIR, "Arduino", "Codegen", "deployed.json")
CODEGEN_HEADER = os.path.join(REPO_DIR, "Arduino", "Codegen", "model_data.h")
INFERENCE_HEADER = os.path.join(
    REPO_DIR, "Arduino", "Inference", "model_data.h"
)
MODEL_INFO_TXT = os.path.join(PY_DIR, "model", "model_info.txt")
DATA_DIR = os.path.join(PY_DIR, "data")

_MAX_BAR_ROWS = 12


def _bar_stylesheet(color_hex: str) -> str:
  """Returns a dark-theme stylesheet for a horizontal probability bar."""
  return (
      "QProgressBar {"
      "  border: 1px solid #3F3F46;"
      "  border-radius: 5px;"
      "  background-color: #18181B;"
      "  height: 22px;"
      "  text-align: right;"
      "}"
      "QProgressBar::chunk {"
      f"  background-color: {color_hex};"
      "  border-radius: 4px;"
      "}"
  )


class _ProbabilityBarRow(QWidget):
  """Single horizontal row showing rank, class label, bar, and percentage."""

  def __init__(self, rank: int, parent: QWidget | None = None) -> None:
    super().__init__(parent)
    lay = QHBoxLayout(self)
    lay.setContentsMargins(2, 2, 2, 2)
    lay.setSpacing(8)

    self.rank_label = QLabel(f"#{rank}")
    self.rank_label.setFixedWidth(28)
    self.rank_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    self.rank_label.setStyleSheet(
        "color: gray; font-family: Menlo, monospace; font-size: 13px;"
    )

    self.name_label = QLabel("—")
    self.name_label.setFixedWidth(105)
    self.name_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    self.name_label.setStyleSheet(AppStyles.get_label_style())

    self.bar = QProgressBar()
    self.bar.setRange(0, 1000)
    self.bar.setValue(0)
    self.bar.setTextVisible(False)
    self.bar.setStyleSheet(_bar_stylesheet("#38BDF8"))

    self.pct_label = QLabel("0.0%")
    self.pct_label.setFixedWidth(64)
    self.pct_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    self.pct_label.setStyleSheet(
        "font-family: Menlo, monospace; font-size: 13px;"
    )

    lay.addWidget(self.rank_label)
    lay.addWidget(self.name_label)
    lay.addWidget(self.bar, 1)
    lay.addWidget(self.pct_label)
    self._last_color = "#38BDF8"

  def update_row(
      self, rank: int, name: str, prob: float, color_hex: str, highlight: bool
  ) -> None:
    """Updates rank text, class name, bar fill, and color."""
    p_clamped = max(0.0, min(1.0, float(prob)))
    self.rank_label.setText(f"#{rank}")
    self.name_label.setText(name)
    if highlight:
      self.name_label.setStyleSheet(
          "font-size: 15px; font-weight: bold;"
      )
      self.pct_label.setStyleSheet(
          "font-family: Menlo, monospace; font-size: 14px; font-weight: bold;"
      )
    else:
      self.name_label.setStyleSheet(AppStyles.get_label_style())
      self.pct_label.setStyleSheet(
          "font-family: Menlo, monospace; font-size: 13px;"
      )
    self.bar.setValue(int(round(p_clamped * 1000.0)))
    self.pct_label.setText(f"{p_clamped * 100.0:5.1f}%")
    if color_hex != self._last_color:
      self._last_color = color_hex
      self.bar.setStyleSheet(_bar_stylesheet(color_hex))


class LiveInferenceWidget(QWidget):
  """Dedicated Live Inference tab with probability bars, IMU plot, and serial log."""

  def __init__(
      self,
      serial_handler: Any = None,
      real_time_graph: Any = None,
      playground_inference_widget: Any = None,
      parent: QWidget | None = None,
  ) -> None:
    super().__init__(parent)
    self.serial_handler = None
    self.real_time_graph = real_time_graph
    self.playground_inference_widget = playground_inference_widget

    self.inference_running = False
    self.log_paused = False
    self.class_names: list[str] = self._discover_class_names()
    self.ema_probs: np.ndarray | None = None
    self.last_raw_probs: list[float] = []
    self.last_latency_us: int | None = None
    self.inference_count = 0
    self.device_info: dict[str, str] = self._load_default_device_info()
    self.recent_gestures: deque[str] = deque(maxlen=8)

    # 8-second 100 Hz rolling IMU buffer for the live acceleration plot
    self.window_seconds = 8
    self.sample_rate = 100
    n_pts = self.window_seconds * self.sample_rate
    self.time_data = np.linspace(0.0, float(self.window_seconds), n_pts)
    self.x_data = np.zeros(n_pts, dtype=float)
    self.y_data = np.zeros(n_pts, dtype=float)
    self.z_data = np.zeros(n_pts, dtype=float)
    self._imu_dirty = False
    self._debug_imu_counter = 0

    self._build_ui()
    self._refresh_device_info_banner()
    self._rebuild_probability_bars()

    if serial_handler is not None:
      self.set_serial_handler(serial_handler)

    self.plot_timer = QTimer(self)
    self.plot_timer.setInterval(33)  # ~30 FPS UI plot refresh
    self.plot_timer.timeout.connect(self._refresh_imu_plot)
    self.plot_timer.start()

  # ---------------------------------------------------------------------------
  # UI Construction
  # ---------------------------------------------------------------------------

  def _build_ui(self) -> None:
    root = QVBoxLayout(self)
    root.setContentsMargins(10, 10, 10, 10)
    root.setSpacing(10)

    # 1. Top Control & Device Telemetry Bar
    top_box = QGroupBox("Device Connection & Live Inference Controls")
    top_box.setStyleSheet(AppStyles.get_group_box_style())
    top_lay = QVBoxLayout(top_box)
    top_lay.setSpacing(8)

    ctrl_row = QHBoxLayout()
    ctrl_row.setSpacing(10)

    self.connect_btn = QPushButton("Connect to Arduino")
    self.connect_btn.setFixedWidth(200)
    self.connect_btn.setStyleSheet(AppStyles.get_button_style())
    self.connect_btn.clicked.connect(self.toggle_connection)
    ctrl_row.addWidget(self.connect_btn)

    self.start_btn = QPushButton("Start Live Inference")
    self.start_btn.setFixedWidth(200)
    self.start_btn.setEnabled(False)
    self.start_btn.setStyleSheet(AppStyles.get_button_style())
    self.start_btn.clicked.connect(self.toggle_inference)
    ctrl_row.addWidget(self.start_btn)

    self.conn_badge = QLabel("Status: Disconnected")
    self.conn_badge.setStyleSheet(AppStyles.get_label_style())
    ctrl_row.addWidget(self.conn_badge)

    self.auto_start_chk = QCheckBox("Auto-start on connect")
    self.auto_start_chk.setChecked(True)
    ctrl_row.addWidget(self.auto_start_chk)

    self.status_btn = QPushButton("Query Device (STATUS)")
    self.status_btn.setToolTip(
        "Send LIVE and STATUS commands to read on-device model metadata"
    )
    self.status_btn.setEnabled(False)
    self.status_btn.setStyleSheet(AppStyles.get_button_style())
    self.status_btn.clicked.connect(self.query_device_status)
    ctrl_row.addWidget(self.status_btn)

    self.reset_stats_btn = QPushButton("Reset Counters")
    self.reset_stats_btn.setStyleSheet(AppStyles.get_button_style())
    self.reset_stats_btn.clicked.connect(self.reset_counters)
    ctrl_row.addWidget(self.reset_stats_btn)

    ctrl_row.addStretch()
    top_lay.addLayout(ctrl_row)

    # Telemetry metadata strip
    self.device_info_label = QLabel()
    self.device_info_label.setStyleSheet(AppStyles.get_note_label_style())
    self.device_info_label.setWordWrap(True)
    top_lay.addWidget(self.device_info_label)
    root.addWidget(top_box)

    # 2. Main Vertical Splitter: Upper (Classification + Bars + IMU Plot) / Lower (Serial Terminal)
    main_splitter = QSplitter(Qt.Vertical, self)

    upper_widget = QWidget()
    upper_lay = QHBoxLayout(upper_widget)
    upper_lay.setContentsMargins(0, 0, 0, 0)
    upper_lay.setSpacing(12)

    # --- Left Column: Hero Prediction Banner + Top-K Probability Bars ---
    left_col = QVBoxLayout()
    left_col.setSpacing(10)

    # Hero Classification Card
    pred_box = QGroupBox("Current Classification")
    pred_box.setStyleSheet(AppStyles.get_group_box_style())
    pred_lay = QVBoxLayout(pred_box)
    pred_lay.setSpacing(6)

    self.hero_frame = QFrame()
    self.hero_frame.setStyleSheet(
        "QFrame { background-color: #2D2D2D; border: 2px solid #52525B;"
        " border-radius: 6px; padding: 8px; }"
    )
    hero_inner = QVBoxLayout(self.hero_frame)
    hero_inner.setContentsMargins(12, 8, 12, 8)
    hero_inner.setSpacing(4)

    self.hero_class_label = QLabel("WAITING FOR STREAM")
    self.hero_class_label.setAlignment(Qt.AlignCenter)
    self.hero_class_label.setStyleSheet(
        "color: #D4D4D8; font-size: 36px; font-weight: bold; border: none;"
    )
    hero_inner.addWidget(self.hero_class_label)

    self.hero_sub_label = QLabel(
        "Connect to Arduino and start live inference to view predictions"
    )
    self.hero_sub_label.setAlignment(Qt.AlignCenter)
    self.hero_sub_label.setStyleSheet(
        "color: #A1A1AA; font-size: 13px; border: none;"
    )
    hero_inner.addWidget(self.hero_sub_label)
    pred_lay.addWidget(self.hero_frame)

    hist_row = QHBoxLayout()
    hist_title = QLabel("Recent Gestures:")
    hist_title.setStyleSheet(AppStyles.get_label_style())
    self.history_label = QLabel("—")
    self.history_label.setStyleSheet(AppStyles.get_label_style())
    self.history_label.setWordWrap(True)
    hist_row.addWidget(hist_title)
    hist_row.addWidget(self.history_label, 1)
    pred_lay.addLayout(hist_row)
    left_col.addWidget(pred_box)

    # Probability Bars GroupBox
    bars_box = QGroupBox("Live Class Probabilities")
    bars_box.setStyleSheet(AppStyles.get_group_box_style())
    bars_lay = QVBoxLayout(bars_box)
    bars_lay.setSpacing(6)

    bars_ctrl = QHBoxLayout()
    bars_ctrl.setSpacing(8)
    lbl_show = QLabel("Show:")
    lbl_show.setStyleSheet(AppStyles.get_label_style())
    bars_ctrl.addWidget(lbl_show)
    self.view_mode_combo = QComboBox()
    self.view_mode_combo.addItems([
        "Top 5 Classes (Sorted)",
        "All Classes (Sorted)",
        "All Classes (Fixed Order)",
    ])
    self.view_mode_combo.setCurrentIndex(0)
    self.view_mode_combo.currentIndexChanged.connect(
        self._on_bar_settings_changed
    )
    bars_ctrl.addWidget(self.view_mode_combo)

    lbl_thresh = QLabel("Conf Threshold:")
    lbl_thresh.setStyleSheet(AppStyles.get_label_style())
    bars_ctrl.addWidget(lbl_thresh)
    self.thresh_spin = QSpinBox()
    self.thresh_spin.setRange(10, 95)
    self.thresh_spin.setSingleStep(5)
    self.thresh_spin.setValue(60)
    self.thresh_spin.setSuffix("%")
    self.thresh_spin.setToolTip(
        "Minimum top-1 probability before a window is marked UNSURE"
    )
    self.thresh_spin.valueChanged.connect(self._on_bar_settings_changed)
    bars_ctrl.addWidget(self.thresh_spin)

    lbl_smooth = QLabel("EMA Smooth:")
    lbl_smooth.setStyleSheet(AppStyles.get_label_style())
    bars_ctrl.addWidget(lbl_smooth)
    self.smooth_spin = QDoubleSpinBox()
    self.smooth_spin.setRange(0.0, 0.85)
    self.smooth_spin.setSingleStep(0.10)
    self.smooth_spin.setDecimals(2)
    self.smooth_spin.setValue(0.00)
    self.smooth_spin.setToolTip(
        "Exponential moving average factor (0.00 = raw per-window probabilities)"
    )
    bars_ctrl.addWidget(self.smooth_spin)
    bars_ctrl.addStretch()
    bars_lay.addLayout(bars_ctrl)

    self.bars_container = QWidget()
    self.bars_container_lay = QVBoxLayout(self.bars_container)
    self.bars_container_lay.setContentsMargins(0, 4, 0, 0)
    self.bars_container_lay.setSpacing(4)

    self.bar_rows: list[_ProbabilityBarRow] = []
    for idx in range(_MAX_BAR_ROWS):
      row_w = _ProbabilityBarRow(idx + 1, self.bars_container)
      self.bars_container_lay.addWidget(row_w)
      self.bar_rows.append(row_w)
    self.bars_container_lay.addStretch()

    bars_lay.addWidget(self.bars_container, 1)
    left_col.addWidget(bars_box, 1)
    upper_lay.addLayout(left_col, 5)

    # --- Right Column: Real-Time 3-Axis IMU Acceleration Plot ---
    imu_box = QGroupBox("Real-Time IMU Acceleration (100 Hz)")
    imu_box.setStyleSheet(AppStyles.get_group_box_style())
    imu_lay = QVBoxLayout(imu_box)
    imu_lay.setSpacing(6)

    imu_header = QHBoxLayout()
    self.imu_stats_label = QLabel(
        "X: +0.00 g   Y: +0.00 g   Z: +0.00 g   |a|: 0.00 g   Peak (1s): 0.00 g"
    )
    self.imu_stats_label.setStyleSheet(AppStyles.get_label_style())
    imu_header.addWidget(self.imu_stats_label, 1)

    self.clear_plot_btn = QPushButton("Clear Trace")
    self.clear_plot_btn.setStyleSheet(AppStyles.get_button_style())
    self.clear_plot_btn.clicked.connect(self.clear_imu_plot)
    imu_header.addWidget(self.clear_plot_btn)
    imu_lay.addLayout(imu_header)

    self.plot_widget = pg.PlotWidget()
    self.plot_widget.setBackground("#2D2D2D")
    self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
    self.plot_widget.enableAutoRange(axis="y")
    self.plot_widget.getViewBox().setAutoVisible(y=True)

    self.x_curve = self.plot_widget.plot(
        pen=pg.mkPen("b", width=2), name="X-Axis"
    )
    self.y_curve = self.plot_widget.plot(
        pen=pg.mkPen("r", width=2), name="Y-Axis"
    )
    self.z_curve = self.plot_widget.plot(
        pen=pg.mkPen("g", width=2), name="Z-Axis"
    )
    self.plot_widget.addLegend(labelTextColor="w")
    self.plot_widget.getAxis("left").setTextPen("w")
    self.plot_widget.getAxis("bottom").setTextPen("w")

    label_style = {
        "color": AppStyles.GRAPH_LABEL_COLOR,
        "font-size": f"{AppStyles.GRAPH_LABEL_FONT_SIZE}pt",
    }
    self.plot_widget.getAxis("left").setLabel(
        "Acceleration", **label_style
    )
    self.plot_widget.getAxis("bottom").setLabel(
        "Time", units="s", **label_style
    )
    tick_font = QFont()
    tick_font.setPointSize(AppStyles.TEXT_FONT_SIZE)
    self.plot_widget.getAxis("bottom").setStyle(tickFont=tick_font)
    self.plot_widget.getAxis("left").setStyle(tickFont=tick_font)

    imu_lay.addWidget(self.plot_widget, 1)
    upper_lay.addWidget(imu_box, 6)

    main_splitter.addWidget(upper_widget)

    # 3. Bottom Panel: Streaming Serial Log Terminal
    term_box = QGroupBox("Serial Monitor")
    term_box.setStyleSheet(AppStyles.get_group_box_style())
    term_lay = QVBoxLayout(term_box)
    term_lay.setSpacing(6)

    term_bar = QHBoxLayout()
    term_bar.setSpacing(8)
    lbl_mode = QLabel("Log Mode:")
    lbl_mode.setStyleSheet(AppStyles.get_label_style())
    term_bar.addWidget(lbl_mode)
    self.log_mode_combo = QComboBox()
    self.log_mode_combo.addItems([
        "Regular (Predictions & Device Info)",
        "Raw Protocol (PROB / LAT / INFO / ERR)",
        "Debug (All Serial + 100 Hz IMU)",
    ])
    self.log_mode_combo.setCurrentIndex(0)
    term_bar.addWidget(self.log_mode_combo)

    self.auto_scroll_chk = QCheckBox("Auto-scroll")
    self.auto_scroll_chk.setChecked(True)
    term_bar.addWidget(self.auto_scroll_chk)

    self.pause_log_btn = QPushButton("Pause Log")
    self.pause_log_btn.setCheckable(True)
    self.pause_log_btn.setStyleSheet(AppStyles.get_button_style())
    self.pause_log_btn.toggled.connect(self._on_pause_log_toggled)
    term_bar.addWidget(self.pause_log_btn)

    self.clear_log_btn = QPushButton("Clear Log")
    self.clear_log_btn.setStyleSheet(AppStyles.get_button_style())
    self.clear_log_btn.clicked.connect(lambda: self.serial_log.clear())
    term_bar.addWidget(self.clear_log_btn)

    term_bar.addStretch()
    lbl_cmd = QLabel("Command:")
    lbl_cmd.setStyleSheet(AppStyles.get_label_style())
    term_bar.addWidget(lbl_cmd)
    self.cmd_input = QLineEdit()
    self.cmd_input.setPlaceholderText("STATUS, LIVE, ...")
    self.cmd_input.setFixedWidth(160)
    self.cmd_input.setStyleSheet(AppStyles.get_line_edit_style())
    self.cmd_input.returnPressed.connect(self._send_custom_command)
    term_bar.addWidget(self.cmd_input)

    self.send_cmd_btn = QPushButton("Send")
    self.send_cmd_btn.setStyleSheet(AppStyles.get_button_style())
    self.send_cmd_btn.clicked.connect(self._send_custom_command)
    term_bar.addWidget(self.send_cmd_btn)
    term_lay.addLayout(term_bar)

    self.serial_log = QPlainTextEdit()
    self.serial_log.setReadOnly(True)
    self.serial_log.setMaximumBlockCount(2000)
    self.serial_log.setStyleSheet(
        "background-color: #1E222A; color: #D8DEE9;"
        " font-family: Menlo, Monaco, monospace; font-size: 11px;"
        " padding: 6px; border-radius: 4px;"
    )
    term_lay.addWidget(self.serial_log, 1)

    main_splitter.addWidget(term_box)
    main_splitter.setStretchFactor(0, 4)
    main_splitter.setStretchFactor(1, 2)
    main_splitter.setSizes([540, 210])

    root.addWidget(main_splitter, 1)

  # ---------------------------------------------------------------------------
  # Class & Device Metadata Discovery
  # ---------------------------------------------------------------------------

  def _discover_class_names(self) -> list[str]:
    """Loads gesture class names from deployed.json, model_data.h, or data/."""
    # 1. Check Arduino/Codegen/deployed.json
    if os.path.isfile(DEPLOYED_JSON):
      try:
        with open(DEPLOYED_JSON, encoding="utf-8") as f:
          d = json.load(f)
        cls_list = d.get("class_names")
        if isinstance(cls_list, list) and cls_list:
          return [str(c) for c in cls_list]
      except Exception:  # pylint: disable=broad-exception-caught
        pass

    # 2. Check Arduino/Codegen/model_data.h or Arduino/Inference/model_data.h
    for hdr in (CODEGEN_HEADER, INFERENCE_HEADER):
      if os.path.isfile(hdr):
        try:
          with open(hdr, encoding="utf-8") as f:
            text = f.read()
          m = re.search(
              r"GESTURES\s*\[[^\]]*\]\s*=\s*\{([^}]+)\}", text, re.DOTALL
          )
          if m:
            names = re.findall(r'"([^"]+)"', m.group(1))
            if names:
              return names
        except Exception:  # pylint: disable=broad-exception-caught
          pass

    # 3. Check data/ directory filenames
    if os.path.isdir(DATA_DIR):
      names_set = set()
      for fname in os.listdir(DATA_DIR):
        if fname.endswith(".txt") and "-" in fname:
          names_set.add(fname.split("-")[0])
      if names_set:
        return sorted(names_set)

    # 4. Check model/model_info.txt
    if os.path.isfile(MODEL_INFO_TXT):
      try:
        with open(MODEL_INFO_TXT, encoding="utf-8") as f:
          for line in f:
            if line.startswith("Classes:"):
              raw = line.split(":", 1)[1].strip()
              return [c.strip() for c in raw.split(",") if c.strip()]
      except Exception:  # pylint: disable=broad-exception-caught
        pass

    return ["Unknown"]

  def _load_default_device_info(self) -> dict[str, str]:
    """Loads initial model telemetry from deployed.json or model_data.h."""
    info: dict[str, str] = {
        "source": "deployed.json (awaiting live device)",
        "model_bytes": "—",
        "window_rows": "100",
        "step": "50",
        "input_type": "int8",
        "output_type": "int8",
        "arena_bytes": "—",
        "solution_id": "—",
        "test_f1": "—",
    }
    if os.path.isfile(DEPLOYED_JSON):
      try:
        with open(DEPLOYED_JSON, encoding="utf-8") as f:
          d = json.load(f)
        tm = d.get("test_metrics", {})
        kb = float(tm.get("model_kb", 0.0))
        if kb > 0:
          info["model_bytes"] = f"{int(round(kb * 1024))} ({kb:.2f} KB)"
        arena_kb = float(tm.get("arena_kb", 0.0))
        if arena_kb > 0:
          info["arena_bytes"] = (
              f"~{int(round(arena_kb * 1024))} B (est {arena_kb:.2f} KB)"
          )
        f1 = float(tm.get("f1_macro", 0.0))
        if f1 > 0:
          info["test_f1"] = f"{f1 * 100.0:.1f}%"
        run_dir = str(d.get("run_dir", ""))
        if run_dir:
          info["solution_id"] = os.path.basename(run_dir)
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    return info

  def _refresh_device_info_banner(self) -> None:
    """Updates the top telemetry strip with current device/model facts."""
    lat_str = (
        f"{self.last_latency_us / 1000.0:.2f} ms ({self.last_latency_us} µs)"
        if self.last_latency_us is not None
        else "—"
    )
    classes_str = ", ".join(self.class_names)
    text = (
        f"Classes ({len(self.class_names)}): [{classes_str}]   |   "
        f"Model: {self.device_info.get('model_bytes', '—')} "
        f"({self.device_info.get('input_type', '—')} -> "
        f"{self.device_info.get('output_type', '—')})   |   "
        f"Arena: {self.device_info.get('arena_bytes', '—')}   |   "
        f"Window: {self.device_info.get('window_rows', '100')} rows "
        f"(step {self.device_info.get('step', '50')})   |   "
        f"MCU Latency: {lat_str}   |   "
        f"Inferences: {self.inference_count}"
    )
    self.device_info_label.setText(text)

  # ---------------------------------------------------------------------------
  # Serial Handler Wiring & Connection Management
  # ---------------------------------------------------------------------------

  def set_serial_handler(self, handler: Any) -> None:
    """Attaches to the shared ArduinoSerialHandler instance."""
    if self.serial_handler is not None:
      for sig_name, slot in (
          ("data_received", self._on_imu_sample),
          ("probability_received", self._on_probabilities),
          ("connection_status", self._on_connection_status),
          ("serial_line_received", self._on_serial_line),
          ("latency_received", self._on_latency),
          ("info_received", self._on_device_info_line),
      ):
        sig = getattr(self.serial_handler, sig_name, None)
        if sig is not None:
          try:
            sig.disconnect(slot)
          except Exception:  # pylint: disable=broad-exception-caught
            pass

    self.serial_handler = handler
    if handler is None:
      return

    handler.data_received.connect(self._on_imu_sample)
    handler.probability_received.connect(self._on_probabilities)
    handler.connection_status.connect(self._on_connection_status)
    if hasattr(handler, "serial_line_received"):
      handler.serial_line_received.connect(self._on_serial_line)
    if hasattr(handler, "latency_received"):
      handler.latency_received.connect(self._on_latency)
    if hasattr(handler, "info_received"):
      handler.info_received.connect(self._on_device_info_line)

    is_conn = bool(getattr(handler, "running", False))
    port_name = getattr(handler, "device_port", None) or "Arduino"
    self._update_connection_ui(
        is_conn, f"Connected to {port_name}" if is_conn else "Disconnected"
    )

  def toggle_connection(self) -> None:
    """Connects or disconnects the Arduino board (synced with RealTimeGraph)."""
    if self.serial_handler is None:
      return

    # Ensure RealTimeGraph is in Arduino mode if the user clicks Connect here
    rtg = self.real_time_graph
    if (
        rtg is not None
        and not getattr(self.serial_handler, "running", False)
        and getattr(rtg, "data_source", 1) != getattr(rtg, "USE_ARDUINO_DATA", 1)
    ):
      if hasattr(rtg, "data_source_checkbox"):
        rtg.data_source_checkbox.setChecked(True)

    if rtg is not None and hasattr(rtg, "toggle_connection"):
      rtg.toggle_connection()
      return

    if getattr(self.serial_handler, "running", False):
      self.serial_handler.disconnect()
    else:
      try:
        self.serial_handler.start()
      except Exception as exc:  # pylint: disable=broad-exception-caught
        QMessageBox.critical(
            self,
            "Connection Error",
            f"Failed to connect to Arduino device: {exc}",
        )

  def _on_connection_status(self, connected: bool, message: str) -> None:
    """Handles connection status changes from ArduinoSerialHandler."""
    self._update_connection_ui(connected, message)
    self._append_log(f"[STATUS] {message}")
    if connected:
      if self.auto_start_chk.isChecked() and not self.inference_running:
        self.start_inference()
      QTimer.singleShot(250, self.query_device_status)
    else:
      if self.inference_running:
        self.stop_inference()

  def _update_connection_ui(self, connected: bool, message: str) -> None:
    self.connect_btn.setText(
        "Disconnect" if connected else "Connect to Arduino"
    )
    self.start_btn.setEnabled(connected)
    self.status_btn.setEnabled(connected)
    if connected:
      self.conn_badge.setText(f"Status: {message}")
    else:
      self.conn_badge.setText(f"Status: {message or 'Disconnected'}")

  def toggle_inference(self) -> None:
    """Starts or stops live inference display."""
    if not self.inference_running:
      if not self.serial_handler or not getattr(
          self.serial_handler, "running", False
      ):
        self._append_log("[WARN] Cannot start inference: Arduino not connected")
        return
      self.start_inference()
    else:
      self.stop_inference()

  def start_inference(self) -> None:
    """Enables live inference processing and syncs with Playground tab."""
    self.inference_running = True
    self.start_btn.setText("Stop Live Inference")
    self.hero_class_label.setText("LISTENING...")
    self.hero_class_label.setStyleSheet(
        "color: #38BDF8; font-size: 36px; font-weight: bold; border: none;"
    )
    self.hero_sub_label.setText("Collecting 1.0 s (100-sample) IMU window...")
    if self.serial_handler and hasattr(self.serial_handler, "send_command"):
      self.serial_handler.send_command("LIVE")

    # Sync with Playground InferenceWidget if present
    piw = self.playground_inference_widget
    if piw is not None and not getattr(piw, "inference_running", False):
      piw.class_names = list(self.class_names)
      piw.start_inference()

  def stop_inference(self) -> None:
    """Pauses live inference processing and syncs with Playground tab."""
    self.inference_running = False
    self.start_btn.setText("Start Live Inference")
    self.hero_class_label.setText("PAUSED")
    self.hero_class_label.setStyleSheet(
        "color: #A1A1AA; font-size: 36px; font-weight: bold; border: none;"
    )
    self.hero_sub_label.setText("Live inference paused")

    piw = self.playground_inference_widget
    if piw is not None and getattr(piw, "inference_running", False):
      piw.stop_inference()

  def query_device_status(self) -> None:
    """Sends LIVE and STATUS commands to query firmware metadata."""
    if self.serial_handler and hasattr(self.serial_handler, "send_command"):
      self.serial_handler.send_command("LIVE")
      ok = self.serial_handler.send_command("STATUS")
      if ok:
        self._append_log("[CMD] Sent LIVE + STATUS to device")

  def reset_counters(self) -> None:
    """Resets inference counter, EMA buffer, and recent gesture history."""
    self.inference_count = 0
    self.ema_probs = None
    self.last_raw_probs = []
    self.recent_gestures.clear()
    self.history_label.setText("—")
    self._refresh_device_info_banner()
    self._rebuild_probability_bars()

  # ---------------------------------------------------------------------------
  # Probability Bar Rendering & Classification Updates
  # ---------------------------------------------------------------------------

  def _on_bar_settings_changed(self) -> None:
    if self.last_raw_probs:
      self._render_probabilities(self.last_raw_probs, count_increment=False)
    else:
      self._rebuild_probability_bars()

  def _rebuild_probability_bars(self) -> None:
    """Initializes empty bars matching current class list and view mode."""
    n_classes = len(self.class_names)
    mode = self.view_mode_combo.currentIndex()
    visible_count = min(5, n_classes) if mode == 0 else min(_MAX_BAR_ROWS, n_classes)
    for idx, row_w in enumerate(self.bar_rows):
      if idx < visible_count:
        cls_name = self.class_names[idx] if idx < n_classes else f"Class {idx}"
        row_w.update_row(idx + 1, cls_name, 0.0, "#38BDF8", highlight=False)
        row_w.setVisible(True)
      else:
        row_w.setVisible(False)

  def _on_latency(self, us: int) -> None:
    self.last_latency_us = us
    self._refresh_device_info_banner()

  def _on_device_info_line(self, line: str) -> None:
    """Parses INFO: and ARENA: lines emitted by Codegen.ino."""
    if line.startswith("INFO:"):
      payload = line[5:].strip()
      if "=" in payload:
        for part in payload.split(","):
          if "=" not in part:
            continue
          k, v = part.split("=", 1)
          k, v = k.strip(), v.strip()
          if k == "classes" and v:
            parsed_classes = [c.strip() for c in v.split("|") if c.strip()]
            if parsed_classes and parsed_classes != self.class_names:
              self.class_names = parsed_classes
              self.ema_probs = None
              self._rebuild_probability_bars()
              if self.playground_inference_widget is not None:
                self.playground_inference_widget.class_names = list(
                    parsed_classes
                )
          elif k == "model_bytes":
            try:
              b_val = int(v)
              self.device_info["model_bytes"] = (
                  f"{b_val:,} B ({b_val / 1024.0:.2f} KB)"
              )
            except ValueError:
              self.device_info["model_bytes"] = v
          else:
            self.device_info[k] = v
    elif line.startswith("ARENA:"):
      val = line[6:].strip()
      try:
        b_val = int(val)
        self.device_info["arena_bytes"] = (
            f"{b_val:,} B ({b_val / 1024.0:.2f} KB)"
        )
      except ValueError:
        self.device_info["arena_bytes"] = val
    self._refresh_device_info_banner()

  def _on_probabilities(self, probabilities: list[float]) -> None:
    """Handles a new PROB: vector from the device."""
    if not probabilities:
      return
    self.last_raw_probs = list(probabilities)
    if not self.inference_running:
      return
    self._render_probabilities(probabilities, count_increment=True)

  def _render_probabilities(
      self, probabilities: list[float], count_increment: bool = True
  ) -> None:
    """Updates the hero classification card, probability bars, and history."""
    n_probs = len(probabilities)
    if n_probs != len(self.class_names):
      # Adapt class names dynamically if device model has a different class count
      if n_probs > len(self.class_names):
        self.class_names = list(self.class_names) + [
            f"Class {i}" for i in range(len(self.class_names), n_probs)
        ]
      else:
        self.class_names = list(self.class_names[:n_probs])
      self.ema_probs = None

    probs_arr = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    alpha = float(self.smooth_spin.value())
    if alpha > 0.0 and self.ema_probs is not None and len(self.ema_probs) == n_probs:
      self.ema_probs = alpha * self.ema_probs + (1.0 - alpha) * probs_arr
    else:
      self.ema_probs = probs_arr.copy()
    disp_probs = self.ema_probs

    if count_increment:
      self.inference_count += 1
      self._refresh_device_info_banner()

    order = np.argsort(-disp_probs)
    top_idx = int(order[0])
    top_prob = float(disp_probs[top_idx])
    top_name = self.class_names[top_idx]
    second_idx = int(order[1]) if n_probs > 1 else top_idx
    second_prob = float(disp_probs[second_idx]) if n_probs > 1 else 0.0
    second_name = self.class_names[second_idx] if n_probs > 1 else "—"
    margin = (top_prob - second_prob) * 100.0

    thresh = float(self.thresh_spin.value()) / 100.0
    is_confident = top_prob >= thresh
    is_null = top_name.lower() == "null"

    if not is_confident:
      accent = "#F59E0B"  # Amber for unsure
      border_bg = "#271F10"
      hero_text = f"UNSURE ({top_name}?)"
    elif is_null:
      accent = "#38BDF8"  # Sky blue for resting / Null
      border_bg = "#0F1E2E"
      hero_text = top_name.upper()
    else:
      accent = "#4ADE80"  # Bright green for confident active gesture
      border_bg = "#072617"
      hero_text = top_name.upper()

    self.hero_frame.setStyleSheet(
        f"QFrame {{ background-color: {border_bg}; border: 2px solid {accent};"
        " border-radius: 8px; padding: 10px; }"
    )
    self.hero_class_label.setText(hero_text)
    self.hero_class_label.setStyleSheet(
        f"color: {accent}; font-size: 40px; font-weight: bold; border: none;"
    )

    lat_part = (
        f"   |   MCU: {self.last_latency_us / 1000.0:.2f} ms"
        if self.last_latency_us is not None
        else ""
    )
    self.hero_sub_label.setText(
        f"Confidence: {top_prob * 100.0:5.1f}%   |   "
        f"Margin vs #{2} ({second_name}): +{margin:4.1f}%{lat_part}"
    )

    if count_increment and is_confident and not is_null:
      self.recent_gestures.appendleft(f"{top_name} ({top_prob * 100.0:.0f}%)")
      self.history_label.setText("  ←  ".join(self.recent_gestures))

    # Update Probability Bar Rows
    mode = self.view_mode_combo.currentIndex()
    if mode == 0:
      indices = list(order[: min(5, n_probs)])
    elif mode == 1:
      indices = list(order[: min(_MAX_BAR_ROWS, n_probs)])
    else:
      indices = list(range(min(_MAX_BAR_ROWS, n_probs)))

    for row_idx, row_w in enumerate(self.bar_rows):
      if row_idx < len(indices):
        cls_i = indices[row_idx]
        p_val = float(disp_probs[cls_i])
        c_name = self.class_names[cls_i]
        is_top = cls_i == top_idx
        if is_top:
          bar_color = accent
        elif p_val >= 0.15:
          bar_color = "#60A5FA"
        else:
          bar_color = "#52525B"
        rank_num = int(np.where(order == cls_i)[0][0]) + 1
        row_w.update_row(
            rank_num, c_name, p_val, bar_color, highlight=is_top
        )
        row_w.setVisible(True)
      else:
        row_w.setVisible(False)

    # Log formatted summary in Regular mode
    if count_increment and self.log_mode_combo.currentIndex() == 0:
      top3 = ", ".join(
          f"{self.class_names[int(i)]}:{disp_probs[int(i)] * 100.0:4.1f}%"
          for i in order[: min(3, n_probs)]
      )
      lat_log = (
          f" | lat={self.last_latency_us / 1000.0:.2f}ms"
          if self.last_latency_us is not None
          else ""
      )
      tag = "UNSURE" if not is_confident else "PRED"
      self._append_log(
          f"[{tag} #{self.inference_count:03d}] {top_name:<9s} "
          f"({top_prob * 100.0:5.1f}%) | top3=[{top3}]{lat_log}"
      )

  # ---------------------------------------------------------------------------
  # Real-Time IMU Acceleration Plot
  # ---------------------------------------------------------------------------

  def _on_imu_sample(self, x: float, y: float, z: float) -> None:
    """Appends a new 100 Hz accelerometer reading to the rolling plot buffer."""
    self.x_data = np.roll(self.x_data, -1)
    self.y_data = np.roll(self.y_data, -1)
    self.z_data = np.roll(self.z_data, -1)
    self.x_data[-1] = x
    self.y_data[-1] = y
    self.z_data[-1] = z
    self._imu_dirty = True

  def _refresh_imu_plot(self) -> None:
    """Repaints the 3-axis acceleration plot at ~30 FPS when new samples arrive."""
    if not self._imu_dirty:
      return
    self._imu_dirty = False
    self.x_curve.setData(self.time_data, self.x_data)
    self.y_curve.setData(self.time_data, self.y_data)
    self.z_curve.setData(self.time_data, self.z_data)

    ax = float(self.x_data[-1])
    ay = float(self.y_data[-1])
    az = float(self.z_data[-1])
    mag = float(np.sqrt(ax * ax + ay * ay + az * az))
    recent_mag = np.sqrt(
        self.x_data[-100:] ** 2
        + self.y_data[-100:] ** 2
        + self.z_data[-100:] ** 2
    )
    peak_1s = float(np.max(recent_mag)) if len(recent_mag) else mag
    self.imu_stats_label.setText(
        f"X: {ax:+5.2f} g   Y: {ay:+5.2f} g   Z: {az:+5.2f} g   "
        f"|a|: {mag:5.2f} g   Peak (1s): {peak_1s:5.2f} g"
    )

  def clear_imu_plot(self) -> None:
    """Zeros the rolling IMU waveform buffer."""
    self.x_data.fill(0.0)
    self.y_data.fill(0.0)
    self.z_data.fill(0.0)
    self._imu_dirty = True
    self._refresh_imu_plot()

  # ---------------------------------------------------------------------------
  # Streaming Serial Monitor
  # ---------------------------------------------------------------------------

  def _on_pause_log_toggled(self, paused: bool) -> None:
    self.log_paused = paused
    self.pause_log_btn.setText("Resume Log" if paused else "Pause Log")

  def _send_custom_command(self) -> None:
    cmd = self.cmd_input.text().strip()
    if not cmd:
      return
    if self.serial_handler and hasattr(self.serial_handler, "send_command"):
      ok = self.serial_handler.send_command(cmd)
      self._append_log(
          f"[TX] {cmd}" if ok else f"[TX ERR] Failed to send '{cmd}' (not connected)"
      )
    self.cmd_input.clear()

  def _on_serial_line(self, line: str) -> None:
    """Filters and streams incoming serial lines according to Log Mode."""
    if self.log_paused:
      return
    mode = self.log_mode_combo.currentIndex()
    if mode == 0:
      # Regular mode: show INFO, ARENA, ERR, UNSURE, READY (PROB is logged with class names)
      if line.startswith(("IMU:", "PROB:", "LAT:", ".")):
        return
      self._append_log(line)
    elif mode == 1:
      # Raw Protocol mode: show all protocol lines except 100 Hz IMU and replay dots
      if line.startswith(("IMU:", ".")):
        return
      self._append_log(line)
    else:
      # Debug mode: show everything including 100 Hz IMU lines
      self._append_log(line)

  def _append_log(self, text: str) -> None:
    if self.log_paused:
      return
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    self.serial_log.appendPlainText(f"[{ts}] {text}")
    if self.auto_scroll_chk.isChecked():
      sb = self.serial_log.verticalScrollBar()
      sb.setValue(sb.maximum())
      self.serial_log.moveCursor(QTextCursor.End)


def attach_live_inference_tab(window: Any) -> LiveInferenceWidget:
  """Attaches the Live Inference tab to MainWindow and links shared serial state."""
  rtg = getattr(window, "real_time_graph", None)
  sh = getattr(rtg, "serial_handler", None) if rtg else None
  piw = getattr(window, "inference_widget", None)

  live_widget = LiveInferenceWidget(
      serial_handler=sh,
      real_time_graph=rtg,
      playground_inference_widget=piw,
      parent=window,
  )
  window.live_inference_widget = live_widget

  if piw is not None:
    orig_start = piw.start_inference
    orig_stop = piw.stop_inference

    def _synced_start() -> None:
      orig_start()
      if not live_widget.inference_running:
        live_widget.start_inference()

    def _synced_stop() -> None:
      orig_stop()
      if live_widget.inference_running:
        live_widget.stop_inference()

    piw.start_inference = _synced_start
    piw.stop_inference = _synced_stop

  tabs = getattr(window, "tabs", None)
  if tabs is not None:
    # Insert right after Playground (index 1) so it sits between Playground and Codegen
    tabs.insertTab(1, live_widget, "Live Inference")
  return live_widget
