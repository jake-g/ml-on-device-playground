"""Motion Synth CLI.

  python -m motion_synth                    window: live board plus speakers
  python -m motion_synth --headless         console only, live board
  python -m motion_synth --headless --record_handling   handling, no throws
  python -m motion_synth --sim all          simulated flights through speakers
  python -m motion_synth --sim throw --wav throw.wav
  python -m motion_synth --replay data/Drop-3.txt --wav drop.wav
  python -m motion_synth --learn            relearn thresholds from data
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import serial
from serial.tools import list_ports

from motion_synth import detector
from motion_synth import synth

# Seeed Studio USB vendor id; the XIAO nRF52840 Sense enumerates with it.
XIAO_VID = 0x2886
# Ports owned by other boards on this bench; never auto-select them.
BLOCKED_PORTS = ("/dev/cu.usbmodem2101", "/dev/cu.usbserial-0001")
BAUD = 115200

logger = logging.getLogger("motion_synth")


def find_xiao_port() -> Optional[str]:
  """First serial port that looks like a XIAO, skipping blocked ports."""
  for port in list_ports.comports():
    if port.device in BLOCKED_PORTS:
      continue
    desc = port.description or ""
    if port.vid == XIAO_VID or "XIAO" in desc or "Seeed" in desc:
      return port.device
  return None


def _parse_triple(line: str,
                  tag: str) -> Optional[Tuple[float, float, float]]:
  if not line.startswith(tag):
    return None
  try:
    x, y, z = (float(v) for v in line[len(tag):].split(","))
  except ValueError:
    return None
  return x, y, z


def parse_imu(line: str) -> Optional[Tuple[float, float, float]]:
  """Parses `IMU:ax,ay,az` (g); returns None for any other line."""
  return _parse_triple(line, "IMU:")


def parse_gyr(line: str) -> Optional[Tuple[float, float, float]]:
  """Parses `GYR:gx,gy,gz` (deg/s); returns None for any other line."""
  return _parse_triple(line, "GYR:")


class SerialReader:
  """Background `IMU:` (and `GYR:`) line reader for headless runs (no Qt)."""

  def __init__(self, callback: Callable[[float, float, float], None],
               port: str,
               gyro_callback: Optional[Callable[[float, float, float],
                                                None]] = None) -> None:
    self.callback = callback
    self.gyro_callback = gyro_callback
    self.port = port
    self._stop = threading.Event()
    self._thread: Optional[threading.Thread] = None

  @property
  def alive(self) -> bool:
    return self._thread is not None and self._thread.is_alive()

  def start(self) -> None:
    """Opens the port and starts reading.

    Raises:
      serial.SerialException: If the port cannot be opened.
    """
    ser = serial.Serial(self.port, BAUD, timeout=0.15, exclusive=True)
    self._thread = threading.Thread(target=self._run, args=(ser,),
                                    daemon=True)
    self._thread.start()

  def stop(self) -> None:
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=1.0)

  def _run(self, ser: serial.Serial) -> None:
    with ser:
      while not self._stop.is_set():
        try:
          raw = ser.readline()
        except serial.SerialException as e:
          logger.error("Serial lost: %s", e)
          return
        line = raw.decode("utf-8", errors="ignore").strip()
        sample = parse_imu(line)
        if sample is not None:
          self.callback(*sample)
          continue
        gyro = parse_gyr(line)
        if gyro is not None and self.gyro_callback is not None:
          self.gyro_callback(*gyro)


def _synth_config(args: argparse.Namespace) -> synth.SynthConfig:
  return synth.SynthConfig(waveform=args.waveform, landing=args.landing,
                           pitch_hz=args.pitch, volume=args.volume,
                           whistle_volume=args.whistle_volume,
                           landing_volume=args.landing_volume,
                           lift_whoosh=not args.no_lift)


def _log_events(eng: synth.Engine) -> None:
  while not eng.events.empty():
    logger.info("%s", synth.describe(eng.events.get()))


def _clip_saver(
    enabled: bool) -> Optional[Callable[[np.ndarray, detector.FlightEvent],
                                        None]]:
  if not enabled:
    return None

  def save(xyz: np.ndarray, event: detector.FlightEvent) -> None:
    logger.info("  saved %s", os.path.basename(synth.save_clip(xyz, event)))

  return save


def run_window(args: argparse.Namespace) -> int:
  """Standalone Qt window with the Motion Synth tab."""
  # Lazy so headless, sim, and learn runs never load Qt.
  # pylint: disable=import-outside-toplevel
  from PyQt5.QtWidgets import QApplication
  from motion_synth import widget
  # pylint: enable=import-outside-toplevel
  app = QApplication(sys.argv[:1])
  win = widget.MotionSynthWidget(autoconnect=not args.no_connect,
                                 synth_config=_synth_config(args))
  win.setWindowTitle("Motion Synth")
  win.resize(1200, 780)
  win.show()
  return app.exec_()


def run_headless(args: argparse.Namespace) -> int:
  """Live board to speakers with events logged to the console."""
  port = args.port or find_xiao_port()
  if port is None:
    logger.error("No XIAO found; plug it in or pass --port")
    return 1
  eng = synth.Engine(
      synth=synth.Synth(synth.default_sample_rate(), _synth_config(args)),
      on_clip=_clip_saver(not args.no_save))
  out = synth.AudioOut(eng.synth)
  reader = SerialReader(eng.feed_live, port, gyro_callback=eng.feed_gyro)
  try:
    out.start()
    reader.start()
  except (RuntimeError, serial.SerialException) as e:
    logger.error("%s", e)
    out.stop()
    return 1
  logger.info("%s", detector.summary(detector.load_report()))
  if args.record_handling:
    eng.start_handling()
    logger.info("Recording handling on %s: carry, pick up, put down, and "
                "swing the ball without throwing it (Ctrl-C saves)", port)
  else:
    logger.info("Listening on %s, playing on %s; throw the ball (Ctrl-C "
                "quits)", port, synth.output_name())
  try:
    while reader.alive:
      _log_events(eng)
      time.sleep(0.05)
  except KeyboardInterrupt:
    pass
  finally:
    reader.stop()
    out.stop()
  if args.record_handling:
    _save_handling(eng)
  return 0


def _save_handling(eng: synth.Engine) -> None:
  seconds = eng.handling_s or 0.0
  path = eng.stop_handling()
  if path is None:
    logger.info("Handling not saved: %.0f s is under %.0f s", seconds,
                synth.MIN_HANDLING_S)
    return
  logger.info("Saved %.0f s of handling with %d false whistles to %s",
              seconds, eng.handling_whistles, path)


def _load_trace(args: argparse.Namespace) -> np.ndarray:
  """Recording (gyro columns kept when present) or simulated trace."""
  if args.replay:
    xyz, gyr = detector.load_imu(args.replay)
    return xyz if gyr is None else np.hstack([xyz, gyr])
  if args.sim == "all":
    return detector.simulate_sequence()
  return detector.simulate(args.sim)


def run_trace(args: argparse.Namespace) -> int:
  """Plays a simulated or recorded trace live, or renders it to a WAV."""
  imu = _load_trace(args)
  if args.wav:
    eng = synth.Engine(synth=synth.Synth(44100, _synth_config(args), seed=0))
    pcm = eng.render_offline(imu)
    synth.write_wav(args.wav, pcm, eng.synth.sample_rate)
    _log_events(eng)
    logger.info("Wrote %s (%.1f s)", args.wav,
                len(pcm) / eng.synth.sample_rate)
    return 0
  eng = synth.Engine(
      synth=synth.Synth(synth.default_sample_rate(), _synth_config(args)))
  out = synth.AudioOut(eng.synth)
  try:
    out.start()
  except RuntimeError as e:
    logger.error("%s (use --wav to render offline)", e)
    return 1
  try:
    dt = 1.0 / detector.STREAM_RATE_HZ
    start = time.monotonic()
    for i, row in enumerate(imu):
      delay = start + i * dt - time.monotonic()
      if delay > 0:
        time.sleep(delay)
      gyro = (row[3], row[4], row[5]) if len(row) >= 6 else None
      eng.feed(row[0], row[1], row[2], gyro=gyro)
      _log_events(eng)
    time.sleep(2.5)  # Let the last landing ring out.
    _log_events(eng)
  except KeyboardInterrupt:
    pass
  finally:
    out.stop()
  return 0


def run_learn(args: argparse.Namespace) -> int:
  """Relearns detector thresholds from `Python/data` and saved flights."""
  config, report = detector.learn_and_save(
      max_false_rate=args.max_false_rate)
  logger.info("%s", detector.summary(report))
  logger.info(
      "free fall < %.2f g for %d samples, impact >= %.1f g, "
      "settle >= %.2f g", config.freefall_g, config.freefall_samples,
      config.impact_g, config.settle_g)
  logger.info(
      "weightless reads %s g, landing peaks %s g (min/median/max), "
      "rest band %s g", report["weightless_g"], report["landing_peak_g"],
      report["rest_band_g"])
  logger.info("false triggers allowed on bare-board gestures: %s",
              ", ".join(report["false_trigger_classes"]) or "none")
  logger.info(
      "in-ball handling: %s whistles/min over %s min (ceiling %s/min); "
      "bounce clips caught %s/%s; %s recordings with gyro",
      report["handling_per_min"], report["handling_minutes"],
      detector.MAX_HANDLING_PER_MIN, report["bounce_hits"],
      report["bounce_clips"], report["gyro_recordings"])
  for label, run in sorted(report["class_max_run"].items()):
    logger.info("  %-11s longest weightless run %2d samples", label, run)
  logger.info("Wrote %s", detector.CALIBRATION_PATH)
  return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
  ap = argparse.ArgumentParser(
      prog="python -m motion_synth",
      description="Motion-driven synthesizer for the XIAO nRF52840 Sense.")
  mode = ap.add_mutually_exclusive_group()
  mode.add_argument("--headless", action="store_true",
                    help="no window: live board, console log")
  mode.add_argument("--sim", choices=detector.KINDS + ("all",),
                    help="play simulated flights instead of the board")
  mode.add_argument("--replay", metavar="FILE",
                    help="play a recorded `ax ay az [gx gy gz]` text file")
  mode.add_argument("--learn", action="store_true",
                    help="relearn thresholds from Python/data and flights")
  ap.add_argument("--wav", metavar="PATH",
                  help="with --sim/--replay: render to a WAV file")
  ap.add_argument("--port", help="serial port for --headless")
  ap.add_argument("--no_connect", action="store_true",
                  help="window: do not auto-connect the board")
  ap.add_argument("--no_save", action="store_true",
                  help="headless: do not save flights for learning")
  ap.add_argument("--record_handling", action="store_true",
                  help="headless: record in-ball handling (no throws) "
                  "until Ctrl-C, for learning")
  ap.add_argument("--max_false_rate", type=float,
                  default=detector.MAX_FALSE_RATE,
                  help="learn: allowed fraction of gestures that trigger "
                  "(0 = strict, slower whistle)")
  ap.add_argument("--waveform", choices=synth.WAVEFORMS,
                  default=synth.WAVEFORM)
  ap.add_argument("--landing", choices=synth.LANDINGS, default=synth.LANDING)
  ap.add_argument("--pitch", type=float, default=synth.PITCH_HZ,
                  help="whistle pitch at release (Hz)")
  ap.add_argument("--volume", type=float, default=synth.VOLUME,
                  help="master volume 0-1")
  ap.add_argument("--whistle_volume", type=float,
                  default=synth.WHISTLE_VOLUME, help="lift and fall 0-1")
  ap.add_argument("--landing_volume", type=float,
                  default=synth.LANDING_VOLUME, help="landing sound 0-1")
  ap.add_argument("--no_lift", action="store_true",
                  help="no rising whoosh when the ball is lifted")
  args = ap.parse_args(argv)
  if args.wav and not (args.sim or args.replay):
    ap.error("--wav needs --sim or --replay")
  if args.record_handling and not args.headless:
    ap.error("--record_handling needs --headless (the window has a toggle)")
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  if args.learn:
    return run_learn(args)
  if args.sim or args.replay:
    return run_trace(args)
  if args.headless:
    return run_headless(args)
  return run_window(args)


if __name__ == "__main__":
  sys.exit(main())
