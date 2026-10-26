"""Motion Synth tab: live IMU in, motion-driven whistle and impact synth out.

Embeds in the playground window next to the Codegen tab, sharing its
`ArduinoSerialHandler` (samples reach the engine on the serial thread via
a direct connection), or runs standalone via `python -m motion_synth`.
"""

from __future__ import annotations

import collections
import dataclasses
import math
import os
import queue
import random
import threading
import time
from typing import Deque, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QCloseEvent, QShowEvent
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QScrollArea, QSlider, QSpinBox, QVBoxLayout, QWidget,
)
import pyqtgraph as pg

from app_styles import AppStyles
from arduino_serial_handler import ArduinoSerialHandler
from motion_synth import detector
from motion_synth import synth

# Seconds of history shown in the plots.
_WINDOW_S = 6.0
# Seconds a BOOM or CAUGHT banner stays up.
_BANNER_HOLD_S = 0.9
_BANNER_COLORS = {
    "ready": "#3B4252",
    "offline": "#4C566A",
    "flying": "#D08770",
    "lifting": "#A3BE8C",
    "bounce": "#EBCB8B",
    "handling": "#B48EAD",
    detector.BOOM: "#BF616A",
    detector.CAUGHT: "#5E81AC",
    detector.TIMEOUT: "#4C566A",
}
_MARKER_COLORS = {
    detector.RELEASE: "#A3BE8C",
    detector.BOOM: "#BF616A",
    detector.CAUGHT: "#5E81AC",
    detector.TIMEOUT: "#4C566A",
}
_TEST_KINDS = ("drop", "throw", "lob", "catch", "bounce", "lift", "putdown")


class MotionSynthWidget(QWidget):
  """Controls, live plots, and flight log for the Motion Synth tab."""

  def __init__(self, serial_handler: Optional[ArduinoSerialHandler] = None,
               autoconnect: bool = False,
               synth_config: Optional[synth.SynthConfig] = None,
               parent: Optional[QWidget] = None) -> None:
    super().__init__(parent)
    self._owns_handler = serial_handler is None
    self.handler = serial_handler or ArduinoSerialHandler()
    self.engine = synth.Engine(
        synth=synth.Synth(synth.default_sample_rate(), synth_config),
        on_clip=self._save_clip)
    self.audio = synth.AudioOut(self.engine.synth)
    # Worker threads post log lines here; the UI timer drains them.
    self._messages: "queue.SimpleQueue[str]" = queue.SimpleQueue()
    self._save_flights = True
    self._stop_sim = threading.Event()
    self._sim_thread: Optional[threading.Thread] = None
    self._markers: List[Tuple[float, pg.InfiniteLine]] = []
    self._pitch: Deque[Tuple[float, float]] = collections.deque(maxlen=400)
    self._banner: Tuple[str, float, Optional[str]] = ("ready", 0.0, None)
    self._banner_color = ""
    self._audio_tried = False
    self._rate_mark = (time.monotonic(), 0, 0)
    self._rate_hz = 0.0
    self._build_ui()
    # GYR: arrives just before its IMU: line on the same thread.
    self.handler.gyro_received.connect(self.engine.feed_gyro,
                                       type=Qt.DirectConnection)
    self.handler.data_received.connect(self.engine.feed_live,
                                       type=Qt.DirectConnection)
    self.handler.connection_status.connect(self._on_connection_status)
    self._timer = QTimer(self)
    self._timer.setInterval(33)
    self._timer.timeout.connect(self._tick)
    self._timer.start()
    app = QApplication.instance()
    if app is not None:
      app.aboutToQuit.connect(self.shutdown)
    if autoconnect and not self.handler.isRunning():
      self.handler.start()

  # -- UI ---------------------------------------------------------------------

  def _build_ui(self) -> None:
    root = QHBoxLayout(self)
    root.setSpacing(AppStyles.WIDGET_SPACING)
    left_widget = QWidget()
    left = QVBoxLayout(left_widget)
    left.setContentsMargins(0, 0, 6, 0)
    left.setSpacing(8)
    left.addWidget(self._device_box())
    left.addWidget(self._synth_box())
    left.addWidget(self._detector_box())
    left.addStretch(1)
    scroll = QScrollArea()
    scroll.setWidget(left_widget)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setFixedWidth(350)
    root.addWidget(scroll, 0)

    right = QVBoxLayout()
    right.setSpacing(8)
    top = QHBoxLayout()
    self.banner = QLabel("READY")
    self.banner.setAlignment(Qt.AlignCenter)
    self.banner.setMinimumHeight(90)
    top.addWidget(self.banner, 3)
    top.addWidget(self._stats_box(), 2)
    right.addLayout(top)

    self.mag_plot = self._make_plot("|a| (g)")
    self.mag_plot.setYRange(0.0, 5.0, padding=0)
    self.mag_curve = self.mag_plot.plot(pen=pg.mkPen("#88C0D0", width=2))
    cfg = self.engine.detector.config
    self.freefall_line = pg.InfiniteLine(
        pos=cfg.freefall_g, angle=0,
        pen=pg.mkPen("#A3BE8C", width=1.5, style=Qt.DashLine))
    self.impact_line = pg.InfiniteLine(
        pos=cfg.impact_g, angle=0,
        pen=pg.mkPen("#BF616A", width=1.5, style=Qt.DashLine))
    self.mag_plot.addItem(self.freefall_line)
    self.mag_plot.addItem(self.impact_line)
    right.addWidget(self._wrap(
        "Acceleration magnitude (green dashed: free fall, red dashed: impact)",
        self.mag_plot), 3)

    self.pitch_plot = self._make_plot("Hz")
    self.pitch_plot.setLogMode(y=True)
    self.pitch_plot.setYRange(math.log10(100.0), math.log10(4000.0),
                              padding=0)
    self.pitch_curve = self.pitch_plot.plot(
        pen=pg.mkPen("#EBCB8B", width=2.5), connect="finite")
    right.addWidget(self._wrap("Whistle pitch", self.pitch_plot), 2)

    self.log = QPlainTextEdit()
    self.log.setReadOnly(True)
    self.log.setMaximumBlockCount(300)
    self.log.setStyleSheet(
        "background-color: #1E222A; color: #D8DEE9;"
        " font-family: Menlo, Monaco, monospace; font-size: 11px;"
        " padding: 6px; border-radius: 4px;")
    right.addWidget(self._wrap("Flight log", self.log), 2)
    root.addLayout(right, 1)
    self._set_banner("ready")
    self._log(detector.summary(detector.load_report()))

  def _wrap(self, title: str, widget: QWidget) -> QGroupBox:
    box = QGroupBox(title)
    box.setStyleSheet(AppStyles.get_group_box_style())
    lay = QVBoxLayout(box)
    lay.addWidget(widget)
    return box

  def _make_plot(self, y_label: str) -> pg.PlotWidget:
    plot = pg.PlotWidget()
    plot.setBackground("#232730")
    plot.showGrid(x=True, y=True, alpha=0.25)
    plot.setXRange(-_WINDOW_S, 0.0, padding=0)
    plot.setMouseEnabled(x=False, y=False)
    style = {"color": AppStyles.GRAPH_LABEL_COLOR, "font-size": "10pt"}
    plot.getAxis("left").setLabel(y_label, **style)
    plot.getAxis("bottom").setLabel("seconds ago", **style)
    return plot

  def _device_box(self) -> QGroupBox:
    box = QGroupBox("Board")
    box.setStyleSheet(AppStyles.get_group_box_style())
    lay = QVBoxLayout(box)
    self.connect_btn = QPushButton("Connect board")
    self.connect_btn.setStyleSheet(AppStyles.get_button_style())
    self.connect_btn.clicked.connect(self._toggle_connection)
    self.status = QLabel("Not connected (Record or Codegen firmware "
                         "streams IMU at 100 Hz)")
    self.status.setWordWrap(True)
    self.status.setStyleSheet(AppStyles.get_note_label_style())
    lay.addWidget(self.connect_btn)
    lay.addWidget(self.status)
    if self.handler.running:
      self.connect_btn.setText("Disconnect")
      self.status.setText("Connected")
    return box

  def _synth_box(self) -> QGroupBox:
    cfg = self.engine.synth.config
    box = QGroupBox("Synth")
    box.setStyleSheet(AppStyles.get_group_box_style())
    lay = QVBoxLayout(box)
    form = QFormLayout()
    self.wave_combo = self._combo(synth.WAVEFORMS, cfg.waveform, "waveform")
    self.landing_combo = self._combo(synth.LANDINGS, cfg.landing, "landing")
    self.pitch_spin = QSpinBox()
    self.pitch_spin.setRange(200, 3000)
    self.pitch_spin.setSingleStep(10)
    self.pitch_spin.setSuffix(" Hz")
    self.pitch_spin.setValue(int(cfg.pitch_hz))
    self.pitch_spin.valueChanged.connect(
        lambda v: setattr(self.engine.synth.config, "pitch_hz", float(v)))
    self.volume_slider = self._slider(cfg.volume, "volume")
    self.whistle_slider = self._slider(cfg.whistle_volume, "whistle_volume")
    self.landing_slider = self._slider(cfg.landing_volume, "landing_volume")
    self.bright_spin = self._double_spin(1.0, 10.0, 0.5, cfg.cutoff_ratio,
                                         " x pitch")
    self.bright_spin.valueChanged.connect(
        lambda v: setattr(self.engine.synth.config, "cutoff_ratio", v))
    self.res_spin = self._double_spin(0.5, 10.0, 0.1, cfg.resonance, " Q")
    self.res_spin.valueChanged.connect(
        lambda v: setattr(self.engine.synth.config, "resonance", v))
    for label, widget in (("Waveform", self.wave_combo),
                          ("Landing sound", self.landing_combo),
                          ("Pitch at release", self.pitch_spin),
                          ("Master volume", self.volume_slider),
                          ("Whistle volume", self.whistle_slider),
                          ("Landing volume", self.landing_slider),
                          ("Filter cutoff", self.bright_spin),
                          ("Resonance", self.res_spin)):
      form.addRow(self._label(label), widget)
    lay.addLayout(form)
    self.sound_check = QCheckBox("Sound on")
    self.sound_check.setChecked(synth.audio_available())
    self.sound_check.toggled.connect(self._set_sound)
    lay.addWidget(self.sound_check)
    self.lift_check = QCheckBox("Rising whoosh when lifted")
    self.lift_check.setChecked(cfg.lift_whoosh)
    self.lift_check.toggled.connect(
        lambda on: setattr(self.engine.synth.config, "lift_whoosh", on))
    lay.addWidget(self.lift_check)
    grid = QGridLayout()
    buttons = [(f"Test {kind.replace('putdown', 'put-down')}",
                lambda _=False, k=kind: self._simulate(k))
               for kind in _TEST_KINDS]
    buttons.append(("Impact", lambda _=False: self._land_now()))
    for i, (text, slot) in enumerate(buttons):
      btn = QPushButton(text)
      btn.setStyleSheet(AppStyles.get_button_style())
      btn.clicked.connect(slot)
      grid.addWidget(btn, i // 2, i % 2)
    lay.addLayout(grid)
    return box

  def _combo(self, items: Tuple[str, ...], value: str,
             field: str) -> QComboBox:
    """Combo box bound to a `SynthConfig` field."""
    combo = QComboBox()
    combo.addItems(items)
    combo.setCurrentText(value)
    combo.currentTextChanged.connect(
        lambda v: setattr(self.engine.synth.config, field, v))
    return combo

  def _slider(self, value: float, field: str) -> QSlider:
    """0 to 100 slider bound to a 0 to 1 `SynthConfig` field."""
    slider = QSlider(Qt.Horizontal)
    slider.setRange(0, 100)
    slider.setValue(int(round(value * 100)))
    slider.valueChanged.connect(
        lambda v: setattr(self.engine.synth.config, field, v / 100.0))
    return slider

  def _detector_box(self) -> QGroupBox:
    cfg = self.engine.detector.config
    box = QGroupBox("Detector")
    box.setStyleSheet(AppStyles.get_group_box_style())
    lay = QVBoxLayout(box)
    form = QFormLayout()
    self.freefall_spin = self._double_spin(0.15, 0.9, 0.05, cfg.freefall_g,
                                           " g")
    self.confirm_spin = QSpinBox()
    self.confirm_spin.setRange(3, 30)
    self.confirm_spin.setSuffix(" samples")
    self.confirm_spin.setValue(cfg.freefall_samples)
    self.impact_spin = self._double_spin(1.5, 12.0, 0.1, cfg.impact_g, " g")
    for widget in (self.freefall_spin, self.confirm_spin, self.impact_spin):
      widget.valueChanged.connect(self._apply_detector)
    form.addRow(self._label("Free fall below"), self.freefall_spin)
    form.addRow(self._label("Confirm after"), self.confirm_spin)
    form.addRow(self._label("Impact above"), self.impact_spin)
    lay.addLayout(form)
    learn_btn = QPushButton("Learn from recorded data")
    learn_btn.setStyleSheet(AppStyles.get_button_style())
    learn_btn.clicked.connect(self._learn)
    lay.addWidget(learn_btn)
    self.handling_btn = QPushButton("Record handling (no throws)")
    self.handling_btn.setCheckable(True)
    self.handling_btn.setStyleSheet(AppStyles.get_button_style())
    self.handling_btn.setToolTip(
        "Carry, pick up, put down, and swing the ball without throwing it. "
        f"Learning keeps whistles on this under "
        f"{detector.MAX_HANDLING_PER_MIN:g} per minute. Saved to "
        f"{detector.HANDLING_DIR}")
    self.handling_btn.toggled.connect(self._toggle_handling)
    lay.addWidget(self.handling_btn)
    self.save_check = QCheckBox("Save flights for learning")
    self.save_check.setChecked(self._save_flights)
    self.save_check.setToolTip(detector.FLIGHTS_DIR)
    self.save_check.toggled.connect(
        lambda on: setattr(self, "_save_flights", on))
    lay.addWidget(self.save_check)
    return box

  def _stats_box(self) -> QGroupBox:
    box = QGroupBox("Last flight")
    box.setStyleSheet(AppStyles.get_group_box_style())
    grid = QGridLayout(box)
    self.stats = {}
    names = ("Airtime", "Launch", "Apex", "Fall", "Impact", "Size")
    for i, name in enumerate(names):
      value = QLabel("-")
      value.setStyleSheet("font-size: 14px; font-weight: bold;")
      grid.addWidget(self._label(name), i % 3, (i // 3) * 2)
      grid.addWidget(value, i % 3, (i // 3) * 2 + 1)
      self.stats[name] = value
    return box

  def _label(self, text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(AppStyles.get_label_style())
    return label

  def _double_spin(self, lo: float, hi: float, step: float, value: float,
                   suffix: str) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(lo, hi)
    spin.setSingleStep(step)
    spin.setDecimals(2)
    spin.setSuffix(suffix)
    spin.setValue(value)
    return spin

  # -- Actions ----------------------------------------------------------------

  def _toggle_connection(self) -> None:
    if self.handler.running or self.handler.isRunning():
      self.handler.disconnect()
    else:
      self.handler.start()

  def _on_connection_status(self, connected: bool, message: str) -> None:
    self.connect_btn.setText("Disconnect" if connected else "Connect board")
    self.status.setText(message or ("Connected" if connected else
                                    "Disconnected"))

  def _set_sound(self, on: bool) -> None:
    if not on:
      self.audio.stop()
      self.engine.synth.silence()
      return
    try:
      self.audio.start()
      self._log(f"Sound on: {synth.output_name()} at "
                f"{self.engine.synth.sample_rate} Hz")
    except RuntimeError as e:
      self._log(f"Sound unavailable: {e}")
      self.sound_check.blockSignals(True)
      self.sound_check.setChecked(False)
      self.sound_check.blockSignals(False)

  def _ensure_sound(self) -> None:
    if not self.audio.running and not self.sound_check.isChecked():
      self.sound_check.setChecked(True)

  def _simulate(self, kind: str) -> None:
    if self._sim_thread is not None and self._sim_thread.is_alive():
      return
    self._ensure_sound()
    trace = detector.simulate(kind, seed=random.randrange(1 << 16))
    self._log(f"Simulating a {kind} (live samples paused)")
    self._stop_sim.clear()
    self._sim_thread = threading.Thread(target=self._play_trace,
                                        args=(trace,), daemon=True)
    self._sim_thread.start()

  def _play_trace(self, trace: np.ndarray) -> None:
    """Feeds a simulated trace at the stream rate (worker thread)."""
    self.engine.sim_active = True
    try:
      dt = 1.0 / detector.STREAM_RATE_HZ
      start = time.monotonic()
      for i, (ax, ay, az) in enumerate(trace):
        if self._stop_sim.is_set():
          break
        delay = start + i * dt - time.monotonic()
        if delay > 0:
          time.sleep(delay)
        self.engine.feed(ax, ay, az)
    finally:
      self.engine.sim_active = False

  def _land_now(self) -> None:
    self._ensure_sound()
    self.engine.synth.land(1.0)
    self._set_banner(detector.BOOM, time.monotonic() + _BANNER_HOLD_S)

  def _apply_detector(self) -> None:
    det = self.engine.detector
    det.config = dataclasses.replace(
        det.config, freefall_g=self.freefall_spin.value(),
        freefall_samples=self.confirm_spin.value(),
        impact_g=self.impact_spin.value())
    self.freefall_line.setValue(det.config.freefall_g)
    self.impact_line.setValue(det.config.impact_g)

  def _learn(self) -> None:
    try:
      config, report = detector.learn_and_save()
    except (OSError, ValueError) as e:
      self._log(f"Learning failed: {e}")
      return
    for widget, value in ((self.freefall_spin, config.freefall_g),
                          (self.confirm_spin, config.freefall_samples),
                          (self.impact_spin, config.impact_g)):
      widget.blockSignals(True)
      widget.setValue(value)
      widget.blockSignals(False)
    self.engine.detector.config = config
    self.freefall_line.setValue(config.freefall_g)
    self.impact_line.setValue(config.impact_g)
    self._log(detector.summary(report))
    self._log(f"  weightless reads {report['weightless_g']} g, landings "
              f"{report['landing_peak_g']} g (min/median/max)")

  def _toggle_handling(self, on: bool) -> None:
    if on:
      self.engine.start_handling()
      self._log("Recording handling: carry, pick up, put down, and swing "
                "the ball without throwing it; press again to save")
      return
    seconds = self.engine.handling_s or 0.0
    path = self.engine.stop_handling()
    self.handling_btn.setText("Record handling (no throws)")
    if path is None:
      self._log(f"Handling not saved: {seconds:.0f} s is under "
                f"{synth.MIN_HANDLING_S:.0f} s")
      return
    self._log(f"Saved {seconds:.0f} s of handling with "
              f"{self.engine.handling_whistles} false whistles to "
              f"{os.path.basename(path)}; press Learn to use it")

  def _save_clip(self, imu: np.ndarray, event: detector.FlightEvent) -> None:
    """Engine callback on the sample thread: keep the flight for learning."""
    if self._save_flights:
      path = synth.save_clip(imu, event)
      self._messages.put(f"  saved {os.path.basename(path)}")

  # -- Refresh ----------------------------------------------------------------

  def _log(self, text: str) -> None:
    self.log.appendPlainText(f"{time.strftime('%H:%M:%S')}  {text}")

  def _set_banner(self, kind: str, until: float = 0.0,
                  text: Optional[str] = None) -> None:
    self._banner = (kind, until, text)
    if text is None:
      text = {
          "ready": "READY",
          "offline": "NO BOARD (try the test buttons)",
          detector.BOOM: f"{self.engine.synth.config.landing.upper()}!",
          detector.CAUGHT: "CAUGHT",
          detector.TIMEOUT: "LOST",
      }.get(kind, kind.upper())
    self._paint_banner(text, _BANNER_COLORS.get(kind, "#3B4252"))

  def _paint_banner(self, text: str, color: str) -> None:
    if self.banner.text() != text:
      self.banner.setText(text)
    if color != self._banner_color:
      self._banner_color = color
      self.banner.setStyleSheet(
          f"background-color: {color}; color: white; font-size: 30px;"
          " font-weight: bold; border-radius: 8px;")

  def _tick(self) -> None:
    now = time.monotonic()
    while not self._messages.empty():
      self._log(self._messages.get())
    while not self.engine.events.empty():
      self._show_event(self.engine.events.get(), now)
    self._update_banner(now)
    self._update_plots(now)
    self._update_rate(now)

  def _show_event(self, event: detector.FlightEvent, now: float) -> None:
    self._log(synth.describe(event))
    t_mark = event.t_release if event.kind == detector.RELEASE else event.t
    line = pg.InfiniteLine(
        pos=t_mark - now, angle=90,
        pen=pg.mkPen(_MARKER_COLORS[event.kind],
                     width=1 if event.bounce else 2))
    self.mag_plot.addItem(line)
    self._markers.append((t_mark, line))
    if event.kind == detector.RELEASE:
      return
    if event.bounce:
      # Keep the throw's stats; the hop only gets a banner.
      if event.kind == detector.BOOM:
        self._set_banner("bounce", now + _BANNER_HOLD_S,
                         f"BOUNCE {event.bounce}")
      return
    self._set_banner(event.kind, now + _BANNER_HOLD_S)
    v0 = event.v0_mps
    self.stats["Airtime"].setText(f"{event.airtime_s:.2f} s")
    self.stats["Launch"].setText(f"{v0:+.1f} m/s")
    self.stats["Apex"].setText(f"{detector.apex_height_m(v0):.2f} m")
    fall = detector.fall_from_apex_m(event.airtime_s, v0)
    self.stats["Fall"].setText(f"{fall:.2f} m")
    self.stats["Impact"].setText(f"{event.peak_g:.1f} g")
    self.stats["Size"].setText(
        f"{event.intensity:.2f}" if event.kind == detector.BOOM else "-")

  def _update_banner(self, now: float) -> None:
    det = self.engine.detector
    handling = self.engine.handling_s
    if handling is not None:
      self.handling_btn.setText(
          f"Stop and save handling ({int(handling) // 60}:"
          f"{int(handling) % 60:02d}, {self.engine.handling_whistles} "
          "whistles)")
    if det.state == detector.FLYING:
      text = (f"HOP {det.bounce}" if det.bounce else
              f"FALLING  {now - det.t_release:.2f} s")
      self._paint_banner(text, _BANNER_COLORS["flying"])
      return
    kind, until, text = self._banner
    if until and now < until:
      self._set_banner(kind, until, text)
      return
    if self.engine.lifting:
      self._paint_banner(f"LIFTING  {det.lift_mps:+.1f} m/s",
                         _BANNER_COLORS["lifting"])
      return
    if handling is not None:
      self._paint_banner("HANDLING (no throws)", _BANNER_COLORS["handling"])
      return
    live = self.handler.running or self.engine.sim_active
    self._set_banner("ready" if live else "offline")

  def _update_plots(self, now: float) -> None:
    t, mag = self.engine.trace()
    keep = t > now - _WINDOW_S
    self.mag_curve.setData(t[keep] - now, np.minimum(mag[keep], 5.0))
    for t_mark, line in list(self._markers):
      if t_mark < now - _WINDOW_S:
        self.mag_plot.removeItem(line)
        self._markers.remove((t_mark, line))
      else:
        line.setValue(t_mark - now)
    hz = self.engine.synth.current_hz
    self._pitch.append((now, hz if hz > 0 else np.nan))
    pts = np.asarray(self._pitch)
    keep = pts[:, 0] > now - _WINDOW_S
    self.pitch_curve.setData(pts[keep, 0] - now, pts[keep, 1])

  def _update_rate(self, now: float) -> None:
    t0, n0, g0 = self._rate_mark
    if now - t0 < 1.0:
      return
    seen, gyro = self.engine.samples_seen, self.engine.gyro_samples
    self._rate_hz = (seen - n0) / (now - t0)
    gyro_hz = (gyro - g0) / (now - t0)
    self._rate_mark = (now, seen, gyro)
    if self.handler.running:
      port = getattr(self.handler.serial_port, "port", None) or "board"
      gyro_text = ("gyro on" if gyro_hz > 0.5 * self._rate_hz > 0 else
                   "no gyro (make flash-synth adds it)")
      self.status.setText(
          f"Connected to {port}, {self._rate_hz:.0f} Hz, {gyro_text}")

  # -- Lifecycle --------------------------------------------------------------

  def showEvent(  # pylint: disable=invalid-name
      self, event: QShowEvent) -> None:
    super().showEvent(event)
    if not self._audio_tried and self.sound_check.isChecked():
      self._audio_tried = True
      self._set_sound(True)

  def closeEvent(  # pylint: disable=invalid-name
      self, event: QCloseEvent) -> None:
    self.shutdown()
    super().closeEvent(event)

  def shutdown(self) -> None:
    """Saves handling in progress; stops audio, simulation, and our board."""
    if self.handling_btn.isChecked():
      self.handling_btn.setChecked(False)
    self._stop_sim.set()
    self.audio.stop()
    if self._owns_handler and (self.handler.running or
                               self.handler.isRunning()):
      self.handler.disconnect()
