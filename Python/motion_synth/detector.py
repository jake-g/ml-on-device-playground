"""Sensing side of Motion Synth: device spec, flight detector, learned thresholds.

An accelerometer reads specific force, so a board in flight reads close to
0 g on the way up as well as on the way down. Real flights on the cable
read 0.2 to 0.5 g because the cable drags and the board spins. A throw or
lift reads above 1 g while the hand pushes, and a landing is a short spike
far above the 0.7 to 1.5 g a board shows at rest or in hand (`Null`).

Sections:
  Spec: LSM6DS3 and stream constants from the Seeed_Arduino_LSM6DS3 2.0.7
    defaults the `Record`, `Codegen`, and `MotionSynth` sketches use.
  Physics: ballistic height, speed, apex, and fall distance.
  Detector: `idle` until `freefall_samples` consecutive samples sit below
    `freefall_g`, then `flying` until an `impact_g` spike (boom),
    `settle_samples` samples back above `settle_g` (caught, often the
    cable going taut), or `max_flight_s` (timeout). A spike shortly after
    a catch is still a (late) boom. For `refractory_s` after a boom, only
    bounces start (a shorter `bounce_samples` confirmation), each with its
    own landing. While idle, the push along the tracked gravity direction
    integrates into a lift speed and height for the rising whoosh: the
    gyro (when the stream has `GYR:` lines) keeps "up" right while the
    board turns, and put-downs, carries, and swings stay quiet.
  Learning: grid-searches the free-fall threshold and confirmation length
    on Drop recordings plus saved flights versus every other gesture
    (within a false-trigger budget) and recorded handling (at
    most `MAX_HANDLING_PER_MIN` triggers per minute), and takes impact and
    settle thresholds from the `Null` rest band.
  Simulation: synthetic drop, lob, throw, catch, bounce, lift, put-down,
    and carry traces shaped like the real recordings.
"""

from __future__ import annotations

import collections
import dataclasses
import glob
import json
import logging
import math
import os
from typing import (Any, Deque, Dict, List, Optional, Sequence, Tuple,
                    Union)

import numpy as np

logger = logging.getLogger(__name__)

Gyro = Optional[Tuple[float, float, float]]

# -- Spec ---------------------------------------------------------------------

# Standard gravity (m/s^2 per g).
G = 9.80665
# Library defaults: +-16 g, 416 Hz ODR, 100 Hz anti-alias; gyro +-2000 dps.
ACCEL_FULL_SCALE_G = 16.0
ACCEL_ODR_HZ = 416.0
ACCEL_BANDWIDTH_HZ = 100.0
GYRO_FULL_SCALE_DPS = 2000.0
# Firmware polls at 100 Hz and prints `IMU:ax,ay,az` with two decimals
# (`MotionSynth` also prints `GYR:gx,gy,gz` in deg/s just before it).
STREAM_RATE_HZ = 100.0
STREAM_RESOLUTION_G = 0.01
# Largest per-axis reading in the dataset: impacts clip here.
AXIS_CLIP_G = 15.99
# |a| with every axis clipped, the hardest hit the stream can report.
MAX_MAGNITUDE_G = math.sqrt(3.0) * AXIS_CLIP_G
# LSM6DS3 FREE_FALL (5Dh) FF_THS choices: the chip's own free-fall engine
# calls anything below 156 to 500 mg weightless.
HW_FREE_FALL_THRESHOLDS_G = (0.156, 0.219, 0.250, 0.312, 0.344, 0.406, 0.469,
                             0.500)

# -- Physics ------------------------------------------------------------------

Number = Union[float, np.ndarray]


def height_m(t_s: Number, v0_mps: float) -> Number:
  """Ballistic height relative to the release point, up is positive."""
  return v0_mps * t_s - 0.5 * G * t_s * t_s


def vertical_speed_mps(t_s: Number, v0_mps: float) -> Number:
  """Vertical velocity since release in m/s, up is positive."""
  return v0_mps - G * t_s


def apex_height_m(v0_mps: float) -> float:
  """Height gained above the release point for an upward launch."""
  return max(v0_mps, 0.0)**2 / (2.0 * G)


def fall_from_apex_m(airtime_s: float, v0_mps: float) -> float:
  """Distance fallen from the apex until the flight ended."""
  t_down = max(airtime_s - max(v0_mps, 0.0) / G, 0.0)
  return 0.5 * G * t_down * t_down


# -- Detector -----------------------------------------------------------------

IDLE = "idle"
FLYING = "flying"
RELEASE = "release"
BOOM = "boom"
CAUGHT = "caught"
TIMEOUT = "timeout"

# Launch speed clamp in m/s; a downward throw integrates negative.
_MAX_UP_MPS = 15.0
_MAX_DOWN_MPS = 8.0
# Airtime that earns the full airtime share of the boom size.
_FULL_AIRTIME_S = 1.2
# Samples kept for launch integration (2.5 s at 100 Hz).
_HISTORY = 256
# A spike this soon after a catch is the ball hitting the floor after the
# cable went taut (seen in the recorded flights): still a boom.
_LATE_LANDING_S = 0.6
# The impact ring-down right after a landing is not a bounce.
_BOUNCE_ARM_S = 0.04
_MAX_BOUNCES = 3
# Lift estimate leaks so it returns to zero once the hand stops pushing.
_LIFT_LEAK_S = 0.3
_LIFT_DOWN_LEAK_S = 1.0
_LIFT_HEIGHT_LEAK_S = 0.6
_MAX_LIFT_MPS = 6.0
# After a landing the impact spike and bounces are not lifts.
_LIFT_COOLDOWN_S = 1.0
# Gravity tracking: trust the accelerometer for "up" only near 1 g, slowly
# when the gyro already follows rotation.
_UP_TRUST_G = 0.1
_UP_GAIN = 0.05
_UP_GAIN_GYRO = 0.01


@dataclasses.dataclass
class DetectorConfig:
  """Detector thresholds; `learn` fits most of them from recordings.

  Attributes:
    freefall_g: |a| below this counts as weightless.
    freefall_samples: Consecutive weightless samples that confirm a flight.
    impact_g: |a| at or above this during a flight is a landing (boom).
    settle_g: |a| at or above this for `settle_samples` ends a flight
      softly (caught).
    settle_samples: Samples above `settle_g` that count as caught.
    bounce_samples: Weightless samples that start a bounce after a boom.
    launch_window_s: Seconds before release integrated for launch speed.
    refractory_s: Seconds after a boom during which only bounces start.
    max_flight_s: Flights longer than this end as a timeout.
    sample_rate_hz: Nominal stream rate used to turn samples into seconds.
  """
  freefall_g: float = 0.50
  freefall_samples: int = 5
  impact_g: float = 2.5
  settle_g: float = 0.75
  settle_samples: int = 12
  bounce_samples: int = 3
  launch_window_s: float = 0.4
  refractory_s: float = 0.9
  max_flight_s: float = 6.0
  sample_rate_hz: float = STREAM_RATE_HZ


@dataclasses.dataclass(frozen=True)
class FlightEvent:
  """One detector transition.

  Attributes:
    kind: `release`, `boom`, `caught`, or `timeout`.
    t: Timestamp of the sample that produced the event in seconds.
    t_release: Back-dated time the ball left the hand (or floor).
    v0_mps: Estimated vertical launch speed, up is positive.
    airtime_s: Seconds weightless (landing events only).
    peak_g: Largest |a| at the landing.
    intensity: Explosion size in [0, 1] (booms only).
    bounce: 0 for a thrown flight, n for the n-th bounce after it.
  """
  kind: str
  t: float
  t_release: float
  v0_mps: float
  airtime_s: float = 0.0
  peak_g: float = 0.0
  intensity: float = 0.0
  bounce: int = 0


def _clip01(x: float) -> float:
  return min(max(x, 0.0), 1.0)


def boom_intensity(peak_g: float, airtime_s: float, impact_g: float) -> float:
  """Explosion size in [0.25, 1] from how hard and how long it fell.

  The hit share is log-scaled between the impact threshold and the
  stream's clipping magnitude (every axis at 16 g).

  Args:
    peak_g: Largest |a| at the landing.
    airtime_s: Seconds weightless before landing.
    impact_g: Impact threshold in g.

  Returns:
    Explosion size.
  """
  span = math.log(MAX_MAGNITUDE_G / impact_g)
  hit = _clip01(math.log(max(peak_g, impact_g) / impact_g) / span)
  return 0.25 + 0.45 * hit + 0.30 * _clip01(airtime_s / _FULL_AIRTIME_S)


class FlightDetector:
  """Turns raw accelerometer (and optional gyro) samples into events."""

  def __init__(self, config: Optional[DetectorConfig] = None) -> None:
    self.config = config or DetectorConfig()
    self.state = IDLE
    self.magnitude_g = 1.0
    # Smoothed |a| while flying (drag plus spin); drives whistle flutter.
    self.flutter_g = 0.0
    # Slow estimate of the at-rest reading, absorbs sensor offset.
    self.rest_g = 1.0
    # Unit "up" in the sensor frame and the reading along it.
    self.up = (0.0, 0.0, 1.0)
    self.vertical_g = 1.0
    # True once any sample arrived with gyro data.
    self.gyro_active = False
    # Push along "up" integrated while idle; drives the rising whoosh.
    self.lift_mps = 0.0
    self.lift_m = 0.0
    self._up_ready = False
    self._history: Deque[Tuple[float, float]] = collections.deque(
        maxlen=_HISTORY)
    self._low_run = 0
    self._settle_run = 0
    self._t_release = 0.0
    self._v0 = 0.0
    self._peak = 0.0
    self._bounce = 0
    # (armed at, until, next bounce number) after a boom.
    self._bounce_window = (-math.inf, -math.inf, 0)
    self._lift_quiet_until = -math.inf
    # (deadline, airtime) of a catch that may still end in a floor hit.
    self._late: Optional[Tuple[float, float]] = None

  @property
  def dt(self) -> float:
    """Nominal seconds per sample."""
    return 1.0 / self.config.sample_rate_hz

  @property
  def t_release(self) -> float:
    """Back-dated release time of the current or last flight."""
    return self._t_release

  @property
  def bounce(self) -> int:
    """Bounce number of the current or last flight, 0 for a throw."""
    return self._bounce

  def reset(self) -> None:
    """Drops any flight in progress and the launch history."""
    self.state = IDLE
    self._history.clear()
    self._low_run = 0
    self._settle_run = 0
    self._bounce_window = (-math.inf, -math.inf, 0)
    self._late = None
    self.lift_mps = self.lift_m = 0.0

  def update(self, t: float, ax: float, ay: float, az: float,
             gyro: Gyro = None) -> Optional[FlightEvent]:
    """Consumes one sample.

    Args:
      t: Sample time in seconds on any monotonic clock.
      ax: X acceleration in g.
      ay: Y acceleration in g.
      az: Z acceleration in g.
      gyro: Optional (gx, gy, gz) rotation rate in deg/s.

    Returns:
      The event this sample triggered, if any.
    """
    mag = math.sqrt(ax * ax + ay * ay + az * az)
    prev = self.magnitude_g
    self.magnitude_g = mag
    self._history.append((t, mag))
    self._track_up(ax, ay, az, mag, gyro)
    if self.state == FLYING:
      return self._update_flying(t, mag, prev)
    if abs(mag - 1.0) < 0.15:
      self.rest_g += 0.01 * (mag - self.rest_g)
    self._update_lift(t)
    if self._late is not None:
      late = self._late_landing(t, mag)
      if late is not None:
        return late
    low = mag < self.config.freefall_g
    armed, until, bounce = self._bounce_window
    if t < until:
      # After a landing only bounces start, on a shorter confirmation.
      if t < armed:
        self._low_run = 0
        return None
      self._low_run = self._low_run + 1 if low else 0
      if self._low_run < self.config.bounce_samples:
        return None
      return self._release(t, bounce)
    self._low_run = self._low_run + 1 if low else 0
    if self._low_run < self.config.freefall_samples:
      return None
    return self._release(t, 0)

  def _track_up(self, ax: float, ay: float, az: float, mag: float,
                gyro: Gyro) -> None:
    """Follows the gravity direction: gyro for turns, 1 g for drift."""
    ux, uy, uz = self.up
    if not self._up_ready and mag > 0.5:
      ux, uy, uz = ax / mag, ay / mag, az / mag
      self._up_ready = True
    if gyro is not None:
      self.gyro_active = True
      wx, wy, wz = (math.radians(w) for w in gyro)
      # A world-fixed vector turns opposite to the body: du = -w x u dt.
      dt = self.dt
      cx, cy, cz = wy * uz - wz * uy, wz * ux - wx * uz, wx * uy - wy * ux
      ux, uy, uz = ux - cx * dt, uy - cy * dt, uz - cz * dt
    if mag > 0.0 and abs(mag - 1.0) < _UP_TRUST_G:
      k = _UP_GAIN_GYRO if gyro is not None else _UP_GAIN
      ux += k * (ax / mag - ux)
      uy += k * (ay / mag - uy)
      uz += k * (az / mag - uz)
    norm = math.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
    self.up = (ux / norm, uy / norm, uz / norm)
    self.vertical_g = (ax * self.up[0] + ay * self.up[1] + az * self.up[2])

  def _update_lift(self, t: float) -> None:
    if t < self._lift_quiet_until:
      self.lift_mps = self.lift_m = 0.0
      return
    # Only the push along "up" counts, so sideways moves stay quiet. Down
    # speed leaks slower, so a put-down's final braking only cancels it.
    v = self.lift_mps + (self.vertical_g - self.rest_g) * G * self.dt
    v *= 1.0 - self.dt / (_LIFT_LEAK_S if v > 0.0 else _LIFT_DOWN_LEAK_S)
    self.lift_mps = min(max(v, -_MAX_LIFT_MPS), _MAX_LIFT_MPS)
    h = self.lift_m + max(self.lift_mps, 0.0) * self.dt
    self.lift_m = h * (1.0 - self.dt / _LIFT_HEIGHT_LEAK_S)

  def _late_landing(self, t: float, mag: float) -> Optional[FlightEvent]:
    deadline, airtime = self._late
    if t > deadline:
      self._late = None
      return None
    if mag < self.config.impact_g:
      return None
    size = boom_intensity(mag, airtime, self.config.impact_g)
    return self._land(BOOM, t, airtime, mag, size)

  def _release(self, t: float, bounce: int) -> FlightEvent:
    """Starts a flight, back-dating the release to the first low sample."""
    cfg = self.config
    hist = list(self._history)
    start = max(len(hist) - self._low_run, 0)
    if bounce:
      v0 = 0.0  # No hand push to integrate.
    else:
      window = int(round(cfg.launch_window_s * cfg.sample_rate_hz))
      push = hist[max(start - window, 0):start]
      # Specific force above rest integrates to vertical launch speed.
      v0 = G * self.dt * sum(m - self.rest_g for _, m in push)
    self._v0 = min(max(v0, -_MAX_DOWN_MPS), _MAX_UP_MPS)
    self._t_release = t - (self._low_run - 0.5) * self.dt
    low = [m for _, m in hist[start:]]
    self.flutter_g = sum(low) / max(len(low), 1)
    self._peak = max(low, default=0.0)
    self._settle_run = 0
    self._late = None
    self._bounce = bounce
    self._bounce_window = (-math.inf, -math.inf, 0)
    self.lift_mps = self.lift_m = 0.0
    self.state = FLYING
    return FlightEvent(RELEASE, t, self._t_release, self._v0, bounce=bounce)

  def _update_flying(self, t: float, mag: float,
                     prev: float) -> Optional[FlightEvent]:
    cfg = self.config
    airtime = t - self._t_release
    self._peak = max(self._peak, mag)
    if mag >= cfg.impact_g:
      peak = max(mag, prev)
      size = boom_intensity(peak, airtime, cfg.impact_g)
      return self._land(BOOM, t, airtime, peak, size)
    self.flutter_g += 0.2 * (mag - self.flutter_g)
    self._settle_run = self._settle_run + 1 if mag >= cfg.settle_g else 0
    if self._settle_run >= cfg.settle_samples:
      airtime -= self._settle_run * self.dt
      return self._land(CAUGHT, t, airtime, self._peak, 0.0)
    if airtime >= cfg.max_flight_s:
      return self._land(TIMEOUT, t, airtime, self._peak, 0.0)
    return None

  def _land(self, kind: str, t: float, airtime: float, peak: float,
            intensity: float) -> FlightEvent:
    cfg = self.config
    bounce = self._bounce
    self.state = IDLE
    self._low_run = 0
    self._late = None
    self._lift_quiet_until = t + _LIFT_COOLDOWN_S
    if kind == BOOM:
      if bounce < _MAX_BOUNCES:
        armed = t + _BOUNCE_ARM_S
      else:
        armed = t + cfg.refractory_s  # Out of bounces: just wait.
      self._bounce_window = (armed, t + cfg.refractory_s, bounce + 1)
    elif kind == CAUGHT and not bounce:
      self._late = (t + _LATE_LANDING_S, max(airtime, 0.0))
    return FlightEvent(kind, t, self._t_release, self._v0, max(airtime, 0.0),
                       peak, intensity, bounce)


# -- Learning -----------------------------------------------------------------

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
# The gesture recordings shared with the playground.
GESTURE_DIR = os.path.join(os.path.dirname(PKG_DIR), "data")
# Flight clips saved from live throws (same text format, optional gyro).
FLIGHTS_DIR = os.path.join(PKG_DIR, "data")
# In-ball handling with no throws (carry, pick up, put down, swing).
HANDLING_DIR = os.path.join(FLIGHTS_DIR, "handling")
# Short flights right after a landing (bounces and bobbles).
BOUNCES_DIR = os.path.join(FLIGHTS_DIR, "bounces")
CALIBRATION_PATH = os.path.join(PKG_DIR, "calibration.json")

FLIGHT_LABEL = "flight"
HANDLING_LABEL = "handling"
BOUNCE_LABEL = "bounce"
POSITIVE_LABELS = ("Drop", FLIGHT_LABEL)
REST_LABEL = "Null"
THRESHOLDS_G = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
CONFIRM_SAMPLES = tuple(range(3, 21))
# Bare-board slaps and flicks may trigger this often: the foam ball rarely
# moves like that, and short-cable flights need a fast whistle.
MAX_FALSE_RATE = 0.07
# Ceiling on whistles per minute of recorded in-ball handling.
MAX_HANDLING_PER_MIN = 1.0
# An impact must clear the top of the hand-held band by this much.
IMPACT_HEADROOM_G = 1.0
# Seconds after a confirmed free fall searched for the landing peak.
LANDING_SEARCH_S = 1.5


@dataclasses.dataclass
class Recording:
  """One labelled recording in g (gyro in deg/s when it was streamed)."""
  name: str
  label: str
  xyz: np.ndarray
  gyr: Optional[np.ndarray] = None

  @property
  def magnitude(self) -> np.ndarray:
    return np.linalg.norm(self.xyz, axis=1)

  @property
  def minutes(self) -> float:
    return len(self.xyz) / STREAM_RATE_HZ / 60.0


def load_imu(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
  """Reads `ax ay az [gx gy gz]` text: acceleration and optional gyro."""
  data = np.loadtxt(path, ndmin=2)
  gyr = data[:, 3:6] if data.shape[1] >= 6 else None
  return data[:, :3], gyr


def load_xyz(path: str) -> np.ndarray:
  """Reads only the acceleration columns of a recording."""
  return load_imu(path)[0]


def load_recordings(gesture_dir: str = GESTURE_DIR,
                    flights_dir: str = FLIGHTS_DIR) -> List[Recording]:
  """Loads gesture recordings, saved flights, handling, and bounces."""
  recs = []
  for path in sorted(glob.glob(os.path.join(gesture_dir, "*.txt"))):
    name = os.path.basename(path)
    recs.append(Recording(name, name[:-4].split("-")[0], load_xyz(path)))
  for sub, label in (("", FLIGHT_LABEL), ("handling", HANDLING_LABEL),
                     ("bounces", BOUNCE_LABEL)):
    for path in sorted(glob.glob(os.path.join(flights_dir, sub, "*.txt"))):
      xyz, gyr = load_imu(path)
      recs.append(Recording(os.path.basename(path), label, xyz, gyr))
  return recs


def _runs_below(magnitude: np.ndarray, threshold: float) -> np.ndarray:
  below = np.concatenate(([0], (magnitude < threshold).astype(np.int8), [0]))
  edges = np.flatnonzero(np.diff(below))
  return edges[1::2] - edges[0::2]


def max_run_below(magnitude: np.ndarray, threshold: float) -> int:
  """Length of the longest run of consecutive samples below `threshold`."""
  runs = _runs_below(magnitude, threshold)
  return int(runs.max()) if runs.size else 0


def count_runs(magnitude: np.ndarray, threshold: float, length: int) -> int:
  """Number of separate runs below `threshold` at least `length` long."""
  return int((_runs_below(magnitude, threshold) >= length).sum())


def landing_peak(magnitude: np.ndarray, threshold: float,
                 confirm: int) -> Optional[float]:
  """Largest |a| after the first confirmed free fall, if there is one."""
  run = 0
  for i, m in enumerate(magnitude):
    run = run + 1 if m < threshold else 0
    if run >= confirm:
      end = i + 1 + int(LANDING_SEARCH_S * STREAM_RATE_HZ)
      tail = magnitude[i + 1:end]
      return float(tail.max()) if tail.size else None
  return None


def _pick_freefall(pos: Sequence[Recording], gestures: Sequence[Recording],
                   handling: Sequence[Recording],
                   max_false_rate: float) -> Dict[str, Any]:
  """Grid-searches the free-fall threshold and confirmation length."""
  allowed = int(math.floor(max_false_rate * len(gestures) + 1e-9))
  minutes = sum(r.minutes for r in handling)
  candidates = []
  for thr in THRESHOLDS_G:
    pos_runs = [max_run_below(r.magnitude, thr) for r in pos]
    neg_runs = {r.name: max_run_below(r.magnitude, thr) for r in gestures}
    for confirm in CONFIRM_SAMPLES:
      false = sorted(n for n, run in neg_runs.items() if run >= confirm)
      if len(false) > allowed:
        continue
      per_min = 0.0
      if minutes:
        hits = sum(count_runs(r.magnitude, thr, confirm) for r in handling)
        per_min = hits / minutes
        if per_min > MAX_HANDLING_PER_MIN:
          continue
      # The shortest feasible confirmation catches the most, soonest.
      candidates.append({
          "freefall_g": thr,
          "freefall_samples": confirm,
          "hits": sum(run >= confirm for run in pos_runs),
          "false": false,
          "handling_per_min": round(per_min, 2),
      })
      break
  if not candidates:
    raise ValueError("no threshold separates free fall from the negatives")

  # Most flights caught, then lowest latency, fewest false triggers, and
  # the more drag-tolerant threshold.
  def rank(c: Dict[str, Any]) -> Tuple[int, int, int, float]:
    return (c["hits"], -c["freefall_samples"], -len(c["false"]),
            c["freefall_g"])

  return max(candidates, key=rank)


def _spread(values: Sequence[float]) -> Optional[List[float]]:
  """Rounded [min, median, max], or None when empty."""
  vals = np.asarray(values, float)
  if vals.size == 0:
    return None
  return [round(float(f(vals)), 2) for f in (np.min, np.median, np.max)]


def learn(
    recordings: Sequence[Recording],
    max_false_rate: float = MAX_FALSE_RATE,
) -> Tuple[DetectorConfig, Dict[str, Any]]:
  """Fits detector thresholds to labelled recordings.

  For each candidate free-fall threshold, the shortest confirmation whose
  false triggers stay within `max_false_rate` of the gesture negatives
  and `MAX_HANDLING_PER_MIN` on recorded handling is kept; the candidate
  catching the most positives wins, then the fastest.

  Args:
    recordings: Gesture recordings plus optional flights, handling, and
      bounces.
    max_false_rate: Allowed fraction of gesture negatives that start a
      whistle (0 gives the strict thresholds).

  Returns:
    The learned config and a JSON-serialisable report of the evidence.

  Raises:
    ValueError: If positives or negatives are missing or inseparable.
  """
  pos = [r for r in recordings if r.label in POSITIVE_LABELS]
  handling = [r for r in recordings if r.label == HANDLING_LABEL]
  bounces = [r for r in recordings if r.label == BOUNCE_LABEL]
  gestures = [r for r in recordings
              if r.label not in POSITIVE_LABELS + (HANDLING_LABEL,
                                                   BOUNCE_LABEL)]
  if not pos or not gestures:
    raise ValueError("need Drop/flight recordings and other gestures")
  best = _pick_freefall(pos, gestures, handling, max_false_rate)
  thr, confirm = best["freefall_g"], best["freefall_samples"]
  config = dataclasses.replace(DetectorConfig(), freefall_g=thr,
                               freefall_samples=confirm)
  rest = [r.magnitude for r in recordings if r.label == REST_LABEL]
  band = None
  if rest:
    mags = np.concatenate(rest)
    band = [round(float(mags.min()), 2), round(float(mags.max()), 2)]
    config.impact_g = round(band[1] + IMPACT_HEADROOM_G, 1)
    config.settle_g = round(band[0] + 0.05, 2)
  weightless = np.concatenate([r.magnitude[r.magnitude < thr] for r in pos])
  peaks = [landing_peak(r.magnitude, thr, confirm) for r in pos]
  flights = [r for r in pos if r.label == FLIGHT_LABEL]
  classes = sorted({r.label for r in recordings})
  report = {
      "recordings": len(recordings),
      "positives": len(pos),
      "flights": len(flights),
      "flight_hits": sum(
          max_run_below(r.magnitude, thr) >= confirm for r in flights),
      "negatives": len(gestures),
      "positive_hits": best["hits"],
      "false_triggers": len(best["false"]),
      "false_trigger_classes": sorted({n.split("-")[0]
                                       for n in best["false"]}),
      "max_false_rate": max_false_rate,
      "handling_minutes": round(sum(r.minutes for r in handling), 2),
      "handling_per_min": best["handling_per_min"],
      "bounce_clips": len(bounces),
      "bounce_hits": sum(
          max_run_below(r.magnitude, thr) >= config.bounce_samples
          for r in bounces),
      "gyro_recordings": sum(r.gyr is not None for r in recordings),
      "latency_ms": round(1000.0 * confirm / STREAM_RATE_HZ),
      "rest_band_g": band,
      "weightless_g": _spread(weightless),
      "landing_peak_g": _spread([p for p in peaks if p is not None]),
      "class_max_run": {
          c: max(max_run_below(r.magnitude, thr)
                 for r in recordings
                 if r.label == c)
          for c in classes
      },
      "spec": {
          "full_scale_g": ACCEL_FULL_SCALE_G,
          "stream_hz": STREAM_RATE_HZ,
          "resolution_g": STREAM_RESOLUTION_G,
          "hw_free_fall_max_g": max(HW_FREE_FALL_THRESHOLDS_G),
      },
  }
  return config, report


def save_calibration(config: DetectorConfig, report: Dict[str, Any],
                     path: str = CALIBRATION_PATH) -> None:
  """Writes the learned config and its evidence as JSON."""
  with open(path, "w", encoding="utf-8") as f:
    json.dump({"config": dataclasses.asdict(config), "report": report}, f,
              indent=2)
    f.write("\n")


def _read_calibration(path: str) -> Dict[str, Any]:
  try:
    with open(path, encoding="utf-8") as f:
      return json.load(f)
  except FileNotFoundError:
    return {}
  except (OSError, ValueError) as e:
    logger.warning("Ignoring unreadable %s: %s", path, e)
    return {}


def load_config(path: str = CALIBRATION_PATH) -> DetectorConfig:
  """Learned config from `path`, or the built-in defaults."""
  names = {f.name for f in dataclasses.fields(DetectorConfig)}
  saved = _read_calibration(path).get("config", {})
  return DetectorConfig(**{k: v for k, v in saved.items() if k in names})


def load_report(path: str = CALIBRATION_PATH) -> Dict[str, Any]:
  """Evidence saved next to the learned config (empty if none)."""
  return _read_calibration(path).get("report", {})


def summary(report: Dict[str, Any]) -> str:
  """One-line human summary of a learn report."""
  if not report:
    return "Using built-in thresholds (no calibration.json yet)."
  text = (f"Learned from {report['recordings']} recordings: free fall "
          f"caught in {report['positive_hits']}/{report['positives']} "
          f"({report.get('flight_hits', 0)}/{report.get('flights', 0)} saved "
          f"flights), bare-board gesture false triggers "
          f"{report['false_triggers']}/{report['negatives']}")
  if report.get("handling_minutes"):
    text += (f", handling {report['handling_per_min']}/min over "
             f"{report['handling_minutes']} min")
  if report.get("bounce_clips"):
    text += (f", bounce clips {report['bounce_hits']}/"
             f"{report['bounce_clips']}")
  return text + f", latency {report['latency_ms']} ms."


def learn_and_save(
    gesture_dir: str = GESTURE_DIR,
    flights_dir: str = FLIGHTS_DIR,
    path: str = CALIBRATION_PATH,
    max_false_rate: float = MAX_FALSE_RATE,
) -> Tuple[DetectorConfig, Dict[str, Any]]:
  """Loads recordings, learns thresholds, and saves them to `path`."""
  config, report = learn(load_recordings(gesture_dir, flights_dir),
                         max_false_rate)
  save_calibration(config, report, path)
  return config, report


# -- Simulation ---------------------------------------------------------------

# kind -> (launch speed m/s, metres landed below the release point).
PRESETS: Dict[str, Tuple[float, float]] = {
    "drop": (0.0, 1.2),
    "lob": (2.5, 1.0),
    "throw": (5.0, 1.0),
    "catch": (3.0, 0.0),
    "bounce": (0.0, 1.2),
}
# Hand motions with no flight: `lift` should whoosh, the others not.
MOTIONS = ("lift", "putdown", "carry")
KINDS = tuple(PRESETS) + MOTIONS
_PUSH_S = 0.18
_UP = np.array([0.0, 0.0, 1.0])


def airtime_s(v0_mps: float, drop_m: float) -> float:
  """Flight time from release until the ball is `drop_m` below release."""
  v0 = max(v0_mps, 0.0)
  return (v0 + math.sqrt(v0 * v0 + 2.0 * G * max(drop_m, 0.0))) / G


def simulate(kind: str = "throw", v0_mps: Optional[float] = None,
             drop_m: Optional[float] = None, impact_g: float = 12.0,
             rate_hz: float = STREAM_RATE_HZ, seed: int = 0) -> np.ndarray:
  """One rest, push, flight, landing, rest trace like the real stream.

  About 1 g at rest, a half-sine push whose area above 1 g integrates to
  the launch speed, a flight reading 0.2 to 0.4 g (drag and spin, as in
  the recordings), then an impact spike, clipped at 15.99 g per axis and
  rounded to the 0.01 g the firmware prints. `bounce` adds two bounces.
  `lift` pushes up and eases to a stop; `putdown` lowers the ball and sets
  it down; `carry` moves it sideways; none of those leave the hand.

  Args:
    kind: One of `KINDS`.
    v0_mps: Launch speed override, up is positive.
    drop_m: Override for how far below the release point it lands.
    impact_g: Peak |a| of the landing spike (ignored for `catch`).
    rate_hz: Output sample rate.
    seed: Noise seed.

  Returns:
    (n, 3) accelerometer samples in g.

  Raises:
    ValueError: For an unknown kind.
  """
  if kind not in KINDS:
    raise ValueError(f"unknown kind {kind!r}; pick from {KINDS}")
  rng = np.random.default_rng(seed)
  rows: List[np.ndarray] = []

  def hold(seconds: float) -> None:
    for _ in range(int(seconds * rate_hz)):
      rows.append(_UP + rng.normal(0.0, 0.012, 3))

  def push(speed: float, seconds: float, axis: np.ndarray = _UP,
           sign: float = 1.0) -> None:
    # Half-sine whose area along `axis` integrates to `speed`.
    n = int(seconds * rate_hz)
    amp = math.pi * speed / (2.0 * G * seconds)
    for i in range(n):
      extra = sign * amp * math.sin(math.pi * (i + 0.5) / n)
      rows.append(_UP + axis * extra + rng.normal(0.0, 0.03, 3))

  def fly(seconds: float, drag: float = 0.25) -> None:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    for i in range(int(seconds * rate_hz)):
      wobble = drag + 0.08 * math.sin(2.0 * math.pi * 3.0 * i / rate_hz)
      rows.append(axis * wobble + rng.normal(0.0, 0.02, 3))

  def impact(peak: float) -> None:
    hit = _UP + 0.3 * rng.normal(size=3)
    hit /= np.linalg.norm(hit)
    for k in (0.3, 1.0, 0.45, 0.2):
      rows.append(hit * peak * k)

  hold(0.8)
  if kind in MOTIONS:
    side = np.array([1.0, 0.0, 0.0])
    if kind == "lift":
      push(1.5, 0.2)
      push(1.5, 0.6, sign=-1.0)
    elif kind == "putdown":
      push(1.2, 0.5, sign=-1.0)
      push(1.2, 0.25)
    else:
      push(1.5, 0.3, axis=side)
      push(1.5, 0.3, axis=side, sign=-1.0)
    hold(1.0)
    return np.round(np.clip(np.asarray(rows), -AXIS_CLIP_G, AXIS_CLIP_G), 2)
  v0 = PRESETS[kind][0] if v0_mps is None else v0_mps
  drop = PRESETS[kind][1] if drop_m is None else drop_m
  if v0 > 0.0:
    push(v0, _PUSH_S)
  fly(airtime_s(v0, drop))
  if kind == "catch":
    for g in (1.5, 2.0, 1.6, 1.3, 1.1):
      rows.append(_UP * g + rng.normal(0.0, 0.03, 3))
  else:
    impact(impact_g)
    if kind == "bounce":
      for k, seconds in enumerate((0.16, 0.08), start=1):
        rows.extend(_UP + rng.normal(0.0, 0.05, 3) for _ in range(6))
        fly(seconds)
        impact(impact_g * 0.45**k)
    for i in range(int(0.25 * rate_hz)):
      ring = 0.5 * math.exp(-i / 6.0) * math.cos(i * 1.3)
      rows.append(_UP * (1.0 + ring) + rng.normal(0.0, 0.03, 3))
  hold(1.0)
  return np.round(np.clip(np.asarray(rows), -AXIS_CLIP_G, AXIS_CLIP_G), 2)


def simulate_sequence(kinds: Sequence[str] = KINDS,
                      seed: int = 0) -> np.ndarray:
  """Back-to-back simulated traces (each already padded with rest)."""
  return np.concatenate(
      [simulate(k, seed=seed + i) for i, k in enumerate(kinds)])
