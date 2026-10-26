"""Sound side of Motion Synth: subtractive synth, flight engine, audio out.

Sections:
  Synth: one basic subtractive `Voice` (`sine`, `triangle`, `square`, `saw`,
    `pulse`, `organ`, `fm`, `wind`, or `noise` oscillator into a resonant RBJ
    low-pass and an amp envelope) plays every sound. The whistle uses the
    selected waveform (`sine` by default): lifting the board raises its pitch
    (the rising whoosh), and in flight the pitch follows ballistic height
    (`semitones_per_m`), so a toss climbs to the apex, then winds down faster
    and faster as it falls. Landing presets: `boom`, `squish`,
    `squelch`, `screech`, `zap`, `boing`, `shatter`, `clap`, and `beep`. The
    whistle and the landing each have their own volume under the master
    volume. Each bounce replays a short hop whistle and a smaller landing.
  Engine: runs samples (plus `GYR:` gyro when the firmware streams it)
    through the detector on whichever thread delivers them, plays the
    events on the synth, keeps plot telemetry, saves each real flight as a
    clip that `detector.learn` can train on, and records handling
    (no throws) that learning holds to a false-whistle ceiling.
  Output: sounddevice stream (optional dependency) and WAV export.
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import math
import os
import queue
import threading
import time
import wave
from typing import (Any, Callable, Deque, List, Optional, Sequence, Tuple,
                    Union)

import numpy as np
from scipy import signal

from motion_synth import detector

try:
  import sounddevice as sd
except (ImportError, OSError) as _err:  # Not installed, or no PortAudio.
  sd = None
  _AUDIO_UNAVAILABLE = (f"sounddevice unavailable ({_err}); "
                        "pip install sounddevice")
else:
  _AUDIO_UNAVAILABLE = ""

logger = logging.getLogger(__name__)

# -- Synth --------------------------------------------------------------------

# Basic settings; the Motion Synth tab exposes these live.
WAVEFORMS = ("sine", "triangle", "square", "saw", "pulse", "organ", "fm",
             "wind")
WAVEFORM = "sine"
LANDINGS = ("boom", "squish", "squelch", "screech", "zap", "boing", "shatter",
            "clap", "beep")
LANDING = "boom"
PITCH_HZ = 1000.0  # Whistle pitch at the release point.
VOLUME = 0.6  # Master.
WHISTLE_VOLUME = 0.8  # Lift whoosh and fall whistle.
LANDING_VOLUME = 1.0  # Landing preset.
CUTOFF_RATIO = 3.0  # Whistle low-pass cutoff as a multiple of its pitch.
RESONANCE = 1.2  # Low-pass Q: 0.707 is flat, higher adds a peak.
SEMITONES_PER_M = 12.0  # One octave of pitch per metre of height.
LIFT_WHOOSH = True  # Rising whistle while the ball is pushed up.

TWO_PI = 2.0 * math.pi
# Samples between filter coefficient updates (control rate).
_CONTROL_BLOCK = 64
_MAX_HITS = 4
# Rough loudness match between waveforms after the low-pass.
_WAVE_TRIM = {
    "sine": 1.0,
    "triangle": 1.1,
    "square": 0.6,
    "saw": 0.75,
    "pulse": 0.75,
    "organ": 0.95,
    "fm": 0.85,
    "wind": 1.05,
    "noise": 1.0,
}
_LIFT = "lift"
_FLIGHT = "flight"

Signal = Union[float, np.ndarray]


@dataclasses.dataclass
class SynthConfig:
  """Synth settings.

  Attributes:
    waveform: Oscillator shape, one of `WAVEFORMS`.
    landing: Landing sound preset, one of `LANDINGS`.
    pitch_hz: Whistle pitch at the release point.
    volume: Master gain in [0, 1].
    whistle_volume: Lift whoosh and fall whistle gain in [0, 1].
    landing_volume: Landing preset gain in [0, 1].
    cutoff_ratio: Whistle low-pass cutoff as a multiple of its pitch.
    resonance: Low-pass Q.
    semitones_per_m: Whistle pitch change per metre of height.
    lift_whoosh: Play the rising whistle while the ball is pushed up.
    ceiling_st: Soft limit on semitones above `pitch_hz`.
    floor_st: Soft limit on semitones below `pitch_hz`.
    vibrato_hz: Vibrato rate.
    vibrato_st: Vibrato depth in semitones with no flutter.
    full_speed_mps: Fall speed at which the whistle reaches full level.
  """
  waveform: str = WAVEFORM
  landing: str = LANDING
  pitch_hz: float = PITCH_HZ
  volume: float = VOLUME
  whistle_volume: float = WHISTLE_VOLUME
  landing_volume: float = LANDING_VOLUME
  cutoff_ratio: float = CUTOFF_RATIO
  resonance: float = RESONANCE
  semitones_per_m: float = SEMITONES_PER_M
  lift_whoosh: bool = LIFT_WHOOSH
  ceiling_st: float = 15.0
  floor_st: float = 30.0
  vibrato_hz: float = 6.0
  vibrato_st: float = 0.12
  full_speed_mps: float = 7.0


def _poly_blep(phase: np.ndarray, inc: np.ndarray) -> np.ndarray:
  """PolyBLEP residual that rounds off a unit step at phase 0."""
  out = np.zeros_like(phase)
  start = phase < inc
  x = phase[start] / inc[start]
  out[start] = x + x - x * x - 1.0
  end = phase > 1.0 - inc
  x = (phase[end] - 1.0) / inc[end]
  out[end] = x * x + x + x + 1.0
  return out


def oscillator(waveform: str, phase: np.ndarray, inc: np.ndarray,
               rng: np.random.Generator) -> np.ndarray:
  """One block of a band-limited waveform.

  Args:
    waveform: One of `WAVEFORMS` or `noise`.
    phase: Phase of each sample in cycles, in [0, 1).
    inc: Phase increment per sample in cycles (frequency / sample rate).
    rng: Noise source.

  Returns:
    Samples in [-1, 1].

  Raises:
    ValueError: For an unknown waveform.
  """
  if waveform == "sine":
    return np.sin(TWO_PI * phase)
  if waveform == "triangle":
    return 4.0 * np.abs(phase - 0.5) - 1.0
  if waveform == "saw":
    return 2.0 * phase - 1.0 - _poly_blep(phase, inc)
  if waveform == "square":
    square = np.where(phase < 0.5, 1.0, -1.0)
    return (square + _poly_blep(phase, inc) -
            _poly_blep((phase + 0.5) % 1.0, inc))
  if waveform == "pulse":
    # Zero-mean 25% pulse built as the difference of two PolyBLEP saws.
    shifted = (phase + 0.25) % 1.0
    saw_a = 2.0 * phase - 1.0 - _poly_blep(phase, inc)
    saw_b = 2.0 * shifted - 1.0 - _poly_blep(shifted, inc)
    return (2.0 / 3.0) * (saw_a - saw_b)
  if waveform == "organ":
    w = TWO_PI * phase
    return (0.35 * np.sin(w) + 0.85 * np.sin(2.0 * w) +
            0.6 * np.sin(3.0 * w)) / 1.45
  if waveform == "fm":
    w = TWO_PI * phase
    index = 1.5 * np.clip(1.0 - 6.0 * inc, 0.0, 1.0)
    return np.sin(2.0 * w + index * np.sin(w))
  if waveform == "wind":
    return 0.65 * np.sin(TWO_PI * phase) + 0.45 * rng.uniform(
        -1.0, 1.0, phase.shape)
  if waveform == "noise":
    return rng.uniform(-1.0, 1.0, phase.shape)
  raise ValueError(f"unknown waveform {waveform!r}")


def lowpass_coefficients(cutoff_hz: float, q: float,
                         sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
  """RBJ cookbook biquad low-pass `(b, a)` for `scipy.signal.lfilter`."""
  w = TWO_PI * min(max(cutoff_hz, 20.0), 0.45 * sample_rate) / sample_rate
  alpha = math.sin(w) / (2.0 * max(q, 0.1))
  cos_w = math.cos(w)
  b = np.array([(1.0 - cos_w) / 2.0, 1.0 - cos_w, (1.0 - cos_w) / 2.0])
  a = np.array([1.0 + alpha, -2.0 * cos_w, 1.0 - alpha])
  return b / a[0], a / a[0]


class Voice:
  """Oscillator -> resonant low-pass -> amp, keeping phase and filter state.

  Pitch, cutoff, and amp may change every sample; the filter coefficients
  follow the cutoff every `_CONTROL_BLOCK` samples.
  """

  def __init__(self, sample_rate: int) -> None:
    self.sample_rate = sample_rate
    self._phase = 0.0
    self._zi = np.zeros(2)

  def render(self, waveform: str, hz: Signal, cutoff_hz: Signal,
             resonance: float, amp: np.ndarray,
             rng: np.random.Generator) -> np.ndarray:
    """Renders `len(amp)` samples.

    Args:
      waveform: Oscillator shape (see `oscillator`).
      hz: Oscillator frequency per sample (or a constant).
      cutoff_hz: Low-pass cutoff per sample (or a constant).
      resonance: Low-pass Q.
      amp: Amplitude envelope per sample.
      rng: Noise source.

    Returns:
      The filtered, enveloped block.
    """
    n = amp.shape[0]
    inc = np.broadcast_to(np.asarray(hz, float) / self.sample_rate, (n,))
    phase = (self._phase + np.cumsum(inc) - inc) % 1.0
    self._phase = float((self._phase + inc.sum()) % 1.0)
    x = oscillator(waveform, phase, inc, rng) * _WAVE_TRIM[waveform]
    cutoff = np.broadcast_to(np.asarray(cutoff_hz, float), (n,))
    y = np.empty(n)
    for s in range(0, n, _CONTROL_BLOCK):
      e = min(s + _CONTROL_BLOCK, n)
      b, a = lowpass_coefficients(cutoff[s], resonance, self.sample_rate)
      y[s:e], self._zi = signal.lfilter(b, a, x[s:e], zi=self._zi)
    return y * amp


def soft_semitones(st: np.ndarray, config: SynthConfig) -> np.ndarray:
  """tanh knees: extreme heights approach, never cross, the pitch limits."""
  up = config.ceiling_st * np.tanh(st / config.ceiling_st)
  down = config.floor_st * np.tanh(st / config.floor_st)
  return np.where(st >= 0.0, up, down)


def whistle_hz(t_s: np.ndarray, v0_mps: float, config: SynthConfig,
               offset_st: float = 0.0) -> np.ndarray:
  """Flight whistle pitch for times since release (height-tracking glide).

  Args:
    t_s: Seconds since release.
    v0_mps: Vertical launch speed, up is positive.
    config: Synth settings.
    offset_st: Semitones carried over from a lift just before release.

  Returns:
    Frequency in Hz for each time.
  """
  height = detector.height_m(np.asarray(t_s, float), v0_mps)
  st = offset_st + config.semitones_per_m * height
  return config.pitch_hz * 2.0**(soft_semitones(st, config) / 12.0)


def whistle_level(t_s: np.ndarray, v0_mps: float,
                  config: SynthConfig) -> np.ndarray:
  """Whistle loudness: steady going up, swelling as the fall speeds up."""
  down = -detector.vertical_speed_mps(np.asarray(t_s, float), v0_mps)
  falling = np.clip(down, 0.0, config.full_speed_mps)
  return 0.35 + 0.65 * falling / config.full_speed_mps


class _Hit:
  """One landing sound (a preset from `LANDINGS`) with its own voices."""

  # preset -> (base seconds, extra seconds at full intensity).
  _LENGTH_S = {
      "boom": (0.5, 2.5),
      "squish": (0.3, 0.4),
      "squelch": (0.45, 0.45),
      "screech": (0.45, 0.55),
      "zap": (0.24, 0.22),
      "boing": (0.6, 0.75),
      "shatter": (0.38, 0.32),
      "clap": (0.25, 0.25),
      "beep": (0.32, 0.0),
  }

  def __init__(self, preset: str, intensity: float, sample_rate: int) -> None:
    if preset not in self._LENGTH_S:
      raise ValueError(f"unknown landing {preset!r}; pick from {LANDINGS}")
    self.preset = preset
    self.intensity = min(max(intensity, 0.0), 1.0)
    self.sample_rate = sample_rate
    self.age = 0
    self._noise = Voice(sample_rate)
    self._tone = Voice(sample_rate)

  @property
  def length(self) -> int:
    base, extra = self._LENGTH_S[self.preset]
    return int(self.sample_rate * (base + extra * self.intensity))

  @property
  def done(self) -> bool:
    return self.age >= self.length

  def render(self, n: int, waveform: str,
             rng: np.random.Generator) -> np.ndarray:
    idx = self.age + np.arange(n)
    self.age += n
    t = idx / self.sample_rate
    fade = np.clip((self.length - idx) / (0.03 * self.sample_rate), 0.0, 1.0)
    preset = getattr(self, f"_{self.preset}")
    return preset(t, waveform, rng) * fade

  def _boom(self, t: np.ndarray, waveform: str,
            rng: np.random.Generator) -> np.ndarray:
    """Noise closing from bright to dark plus a pitch-dropping sub thump."""
    i = self.intensity
    sr = self.sample_rate
    cutoff = 140.0 + (2500.0 + 7500.0 * i) * np.exp(-t / (0.05 + 0.25 * i))
    # Darker noise carries less power; lift the tail so it stays audible.
    lift = (sr / (TWO_PI * cutoff))**0.3
    env = np.minimum(t / 0.002, 1.0) * np.exp(-t / (0.12 + 0.6 * i))
    noise = self._noise.render("noise", 0.0, cutoff, 0.9,
                               3.5 * i**0.75 * lift * env, rng)
    # Thump glides from the low mids (audible on laptop speakers) down.
    hz = 50.0 + (70.0 + 110.0 * i) * np.exp(-t / 0.08)
    thump_amp = 1.2 * i**0.7 * np.exp(-t / (0.1 + 0.25 * i))
    return noise + self._tone.render(waveform, hz, 4.0 * hz, 1.0, thump_amp,
                                     rng)

  def _squish(self, t: np.ndarray, waveform: str,
              rng: np.random.Generator) -> np.ndarray:
    """Single wet filter dive on noise plus a dropping bloop."""
    i = self.intensity
    cutoff = 250.0 + 2600.0 * np.exp(-t / 0.07)
    env = np.minimum(t / 0.005, 1.0) * np.exp(-t / (0.07 + 0.1 * i))
    noise = self._noise.render("noise", 0.0, cutoff, 6.0,
                               0.5 * i**0.6 * env, rng)
    hz = 70.0 + 330.0 * np.exp(-t / 0.06)
    bloop_amp = 0.9 * i**0.6 * np.exp(-t / 0.1)
    return noise + self._tone.render(waveform, hz, 3.0 * hz, 2.0, bloop_amp,
                                     rng)

  def _squelch(self, t: np.ndarray, waveform: str,
               rng: np.random.Generator) -> np.ndarray:
    """Bubbly acid squelch: high-Q filter wobbling across a sticky saw."""
    del waveform
    i = self.intensity
    wobble = 0.5 + 0.5 * np.sin(TWO_PI * (13.0 + 5.0 * i) * t)
    hz = (95.0 + 260.0 * np.exp(-t / 0.18)) * (1.0 + 0.35 * wobble)
    cutoff = 240.0 + (1800.0 + 1800.0 * i) * wobble * np.exp(
        -t / (0.18 + 0.22 * i))
    env = np.minimum(t / 0.003, 1.0) * np.exp(-t / (0.15 + 0.2 * i))
    gain = i**0.6 * env
    tone = self._tone.render("saw", hz, cutoff, 8.0, 0.85 * gain, rng)
    wet = self._noise.render("noise", 0.0, 1.3 * cutoff, 6.0, 0.5 * gain, rng)
    return tone + wet

  def _screech(self, t: np.ndarray, waveform: str,
               rng: np.random.Generator) -> np.ndarray:
    """Metallic skid screech: harsh FM glide plus resonant friction noise."""
    del waveform
    i = self.intensity
    grit = 1.0 + 0.09 * np.sin(TWO_PI * 38.0 * t) + 0.05 * np.sin(
        TWO_PI * 61.0 * t)
    hz = (720.0 + (1500.0 + 900.0 * i) * np.exp(-t / (0.18 + 0.25 * i))) * grit
    env = np.minimum(t / 0.004, 1.0) * np.exp(-t / (0.15 + 0.24 * i))
    gain = i**0.6 * env
    tone = self._tone.render("fm", hz, 2.2 * hz, 4.5, 0.75 * gain, rng)
    skid = self._noise.render("noise", 0.0, 1.5 * hz, 7.0, 0.6 * gain, rng)
    return tone + skid

  def _zap(self, t: np.ndarray, waveform: str,
           rng: np.random.Generator) -> np.ndarray:
    """Ray-gun pew: fast exponential pitch plunge with a bright snap."""
    del waveform
    i = self.intensity
    hz = 65.0 + (2600.0 + 1400.0 * i) * np.exp(-t / (0.065 + 0.055 * i))
    env = np.minimum(t / 0.0015, 1.0) * np.exp(-t / (0.07 + 0.08 * i))
    pew = self._tone.render("saw", hz, 3.5 * hz, 3.5, 1.05 * i**0.6 * env, rng)
    snap = self._noise.render("noise", 0.0, 4500.0, 2.0,
                              0.6 * i**0.6 * np.exp(-t / 0.012), rng)
    return pew + snap

  def _boing(self, t: np.ndarray, waveform: str,
             rng: np.random.Generator) -> np.ndarray:
    """Cartoon spring: twangy pulse oscillating up and down in pitch."""
    del waveform
    i = self.intensity
    spring = np.sin(TWO_PI * 11.5 * t) * np.exp(-t / (0.22 + 0.25 * i))
    hz = (190.0 + 90.0 * i) * 2.0**((5.0 + 4.0 * i) * spring / 12.0)
    cutoff = 300.0 + (700.0 + 1800.0 * (0.5 + 0.5 * spring)) * np.exp(
        -t / (0.3 + 0.3 * i))
    env = np.minimum(t / 0.003, 1.0) * np.exp(-t / (0.18 + 0.24 * i))
    return self._tone.render("pulse", hz, cutoff, 4.5, 1.0 * i**0.6 * env, rng)

  def _shatter(self, t: np.ndarray, waveform: str,
               rng: np.random.Generator) -> np.ndarray:
    """Crisp staggered noise shards plus stepped glassy FM pings."""
    del waveform
    i = self.intensity
    shards = np.zeros_like(t)
    hz = np.full_like(t, 2900.0)
    hits = ((0.0, 1.0, 2900.0), (0.027, 0.75, 2150.0), (0.061, 0.6, 3420.0),
            (0.105, 0.45, 1820.0), (0.16, 0.35, 2580.0))
    for start, amp_k, shard_hz in hits:
      on = t >= start
      shards += amp_k * on * np.exp(
          -np.maximum(t - start, 0.0) / (0.016 + 0.022 * i))
      hz = np.where(on, shard_hz, hz)
    gain = i**0.6 * shards
    spray = self._noise.render("noise", 0.0, 4400.0, 3.0, 1.05 * gain, rng)
    ring = self._tone.render("fm", hz, 5600.0, 2.5, 0.65 * gain, rng)
    return spray + ring

  def _clap(self, t: np.ndarray, waveform: str,
            rng: np.random.Generator) -> np.ndarray:
    """Three quick noise bursts and a short tail through a bright low-pass."""
    del waveform  # Pure noise.
    i = self.intensity
    env = np.zeros_like(t)
    for start in (0.0, 0.011, 0.022):
      env += (t >= start) * np.exp(-np.maximum(t - start, 0.0) / 0.004)
    tail = np.exp(-np.maximum(t - 0.03, 0.0) / (0.06 + 0.12 * i))
    env += 0.6 * (t >= 0.03) * tail
    return self._noise.render("noise", 0.0, 2600.0, 1.8, 1.4 * i**0.5 * env,
                              rng)

  def _beep(self, t: np.ndarray, waveform: str,
            rng: np.random.Generator) -> np.ndarray:
    """Two falling beeps in the selected waveform."""
    i = self.intensity
    first = np.clip(t / 0.004, 0, 1) * np.clip((0.11 - t) / 0.004, 0, 1)
    second = (np.clip((t - 0.16) / 0.004, 0, 1) *
              np.clip((0.27 - t) / 0.004, 0, 1))
    hz = np.where(t < 0.135, 1320.0, 880.0)
    amp = 0.5 * (0.4 + 0.6 * i) * (first + second)
    return self._tone.render(waveform, hz, 5.0 * hz, 0.9, amp, rng)


@dataclasses.dataclass(frozen=True)
class _Flight:
  """Whistle shape for one flight, thrown or bounced.

  Attributes:
    t_release: Clock time the ball left the hand or floor.
    v0_mps: Vertical launch speed, up is positive.
    offset_st: Semitones the pitch starts from (lift carry or floor).
    level: Fixed loudness, or None to swell with fall speed.
    pitch_gain: Multiplier on `semitones_per_m` (exaggerates small hops).
  """
  t_release: float = 0.0
  v0_mps: float = 0.0
  offset_st: float = 0.0
  level: Optional[float] = None
  pitch_gain: float = 1.0


class Synth:
  """One whistle (lift or flight) plus up to four overlapping landings.

  Control methods may be called from any thread; `render` belongs to the
  audio callback (or an offline loop) and owns all DSP state.
  """

  def __init__(self, sample_rate: int = 44100,
               config: Optional[SynthConfig] = None,
               seed: Optional[int] = None) -> None:
    self.sample_rate = int(sample_rate)
    self.config = config or SynthConfig()
    # Smoothed |a| in flight (drag plus spin), set by the engine.
    self.flutter_g = 0.0
    # Whistle pitch at the end of the last block, 0 when silent.
    self.current_hz = 0.0
    self._rng = np.random.default_rng(seed)
    self._lock = threading.Lock()
    # Shared control state, guarded by `_lock`.
    self._mode: Optional[str] = None
    self._flight = _Flight()
    self._lift_st = 0.0
    self._snap_st = False
    self._target = 0.0
    self._tau = 0.01
    self._generation = 0
    self._pending: List[float] = []
    self._boost = 0.0
    # Audio-thread state.
    self._gain = 0.0
    self._st = 0.0
    self._whistle_voice = Voice(self.sample_rate)
    self._hits: List[_Hit] = []

  def lift(self, st: float) -> None:
    """Drives the whistle from hand motion, `st` semitones above pitch."""
    with self._lock:
      if self._mode == _FLIGHT:
        return
      if self._mode is None:
        self._snap_st = True
        self._generation += 1
      self._mode = _LIFT
      self._lift_st = st
      self._target, self._tau = 0.7, 0.02

  def start_whistle(self, t_release: float, v0_mps: float,
                    offset_st: float = 0.0, level: Optional[float] = None,
                    pitch_gain: float = 1.0) -> None:
    """Starts the flight whistle for a ball released at `t_release`.

    Args:
      t_release: Clock time the ball left the hand or floor.
      v0_mps: Vertical launch speed, up is positive.
      offset_st: Semitones above `pitch_hz` the whistle starts from.
      level: Fixed loudness in [0, 1], or None to swell with fall speed.
      pitch_gain: Multiplier on `semitones_per_m` for this flight.
    """
    with self._lock:
      self._mode = _FLIGHT
      self._flight = _Flight(t_release, v0_mps, offset_st, level, pitch_gain)
      self._target, self._tau = 1.0, 0.012
      self._generation += 1

  def stop_whistle(self, fade_s: float = 0.15) -> None:
    """Fades the whistle out over roughly `fade_s` seconds."""
    with self._lock:
      self._target, self._tau = 0.0, max(fade_s, 0.004) / 4.0

  def land(self, intensity: float) -> None:
    """Queues the landing preset at size `intensity` in [0, 1]."""
    with self._lock:
      self._pending.append(float(intensity))

  def boost(self, intensity: float) -> None:
    """Raises the newest landing if it started less than 60 ms ago."""
    with self._lock:
      self._boost = max(self._boost, float(intensity))

  def silence(self) -> None:
    """Drops every voice at once."""
    with self._lock:
      self._mode = None
      self._target = 0.0
      self._pending = []
      self._boost = 0.0
    self._gain = 0.0
    self._hits = []
    self.current_hz = 0.0

  def render(self, frames: int, now: float) -> np.ndarray:
    """Renders the next block.

    Args:
      frames: Number of samples to render.
      now: Clock time of the first sample, same clock as `t_release`.

    Returns:
      Mono float32 samples in [-1, 1].
    """
    cfg = self.config
    with self._lock:
      mode, flight, lift_st = self._mode, self._flight, self._lift_st
      target, tau, generation = self._target, self._tau, self._generation
      snap, self._snap_st = self._snap_st, False
      pending, self._pending = self._pending, []
      boost, self._boost = self._boost, 0.0
    if snap:
      self._st = lift_st
    self._hits.extend(_Hit(cfg.landing, i, self.sample_rate) for i in pending)
    del self._hits[:-_MAX_HITS]
    newest = self._hits[-1] if self._hits else None
    if boost and newest and newest.age < 0.06 * self.sample_rate:
      newest.intensity = max(newest.intensity, min(boost, 1.0))
    out = np.zeros(frames)
    if mode is None:
      self.current_hz = 0.0
    else:
      out += cfg.whistle_volume * self._whistle(
          frames, now, mode, flight, lift_st, target, tau, generation)
    landing = np.zeros(frames)
    for hit in self._hits:
      landing += hit.render(frames, cfg.waveform, self._rng)
    self._hits = [h for h in self._hits if not h.done]
    out += cfg.landing_volume * landing
    return (np.tanh(out) * cfg.volume).astype(np.float32)

  def _whistle(self, n: int, now: float, mode: str, flight: _Flight,
               lift_st: float, target: float, tau: float,
               generation: int) -> np.ndarray:
    cfg = self.config
    sr = self.sample_rate
    if mode == _FLIGHT:
      t = np.maximum(now - flight.t_release + np.arange(n) / sr, 0.0)
      st = flight.offset_st + (flight.pitch_gain * cfg.semitones_per_m *
                               detector.height_m(t, flight.v0_mps))
      if flight.level is None:
        level = whistle_level(t, flight.v0_mps, cfg)
      else:
        level = np.full(n, flight.level)
      flutter = min(max(self.flutter_g, 0.0), 0.8)
    else:
      # Lift: glide toward the latest target (30 ms smoothing).
      t = now + np.arange(n) / sr
      end = self._st + (lift_st - self._st) * (1.0 - math.exp(-n /
                                                             (sr * 0.03)))
      st = np.linspace(self._st, end, n)
      level = np.ones(n)
      flutter = 0.0
    self._st = float(st[-1])
    depth = cfg.vibrato_st * (1.0 + 3.0 * flutter)
    vibrato = 2.0**(depth * np.sin(TWO_PI * cfg.vibrato_hz * t) / 12.0)
    hz = cfg.pitch_hz * 2.0**(soft_semitones(st, cfg) / 12.0) * vibrato
    g0 = self._gain
    g1 = target + (g0 - target) * math.exp(-n / (sr * tau))
    self._gain = g1
    amp = 0.6 * np.linspace(g0, g1, n) * level
    out = self._whistle_voice.render(cfg.waveform, hz, cfg.cutoff_ratio * hz,
                                     cfg.resonance, amp, self._rng)
    self.current_hz = float(hz[-1]) if g1 > 0.02 else 0.0
    if target == 0.0 and g1 < 1e-4:
      with self._lock:
        if self._generation == generation:
          self._mode = None
    return out


# -- Engine -------------------------------------------------------------------

# Plot history: 8 s at 100 Hz.
_TRACE_LEN = 800
# Raw samples for flight clips: pre-roll, longest flight, and post-roll.
_RAW_LEN = 1000
# Samples after a landing trigger still searched for the true impact peak.
_PEAK_WATCH_SAMPLES = 3
PRE_ROLL_S = 1.0
POST_ROLL_S = 0.6
# A soft catch gets a tiny version of the landing preset.
CATCH_PUFF = 0.06
LANDING_FADE_S = 0.01
CATCH_FADE_S = 0.15
# Lift whoosh: starts above this lift speed, ends below the lower one.
LIFT_ON_MPS = 0.8
LIFT_OFF_MPS = 0.25
# Pitch rise per m/s of push plus exaggerated rise height (hand rises are
# only tens of centimetres).
LIFT_ST_PER_MPS = 5.0
LIFT_HEIGHT_GAIN = 2.0
LIFT_FADE_S = 0.15
# A release this soon after a lift continues from the lift's top pitch.
LIFT_CARRY_S = 0.5
# Bounce n plays its hop whistle and landing at BOUNCE_DECAY**n of full
# size. A hop starts from the pitch where the throw met the floor, with an
# assumed rebound speed and its few centimetres of height exaggerated so
# the arc is audible.
BOUNCE_DECAY = 0.5
BOUNCE_V0_MPS = 1.5
BOUNCE_PITCH_GAIN = 4.0
# Handling recordings shorter than this are not saved.
MIN_HANDLING_S = 5.0

ClipCallback = Callable[[np.ndarray, detector.FlightEvent], None]
# One stream sample: ax, ay, az in g and gyro in deg/s (None without GYR:).
Row = Tuple[float, float, float, detector.Gyro]


def describe(event: detector.FlightEvent) -> str:
  """Human-readable one-liner for a flight event."""
  if event.bounce:
    if event.kind == detector.RELEASE:
      return f"bounce {event.bounce}: hop"
    hop_cm = 100.0 * detector.apex_height_m(0.5 * detector.G *
                                            event.airtime_s)
    text = f"bounce {event.bounce} {event.kind}: hop ~{hop_cm:.0f} cm"
    if event.kind == detector.BOOM:
      text = (f"BOUNCE {event.bounce} {event.intensity:.2f}: hop "
              f"~{hop_cm:.0f} cm, impact {event.peak_g:.1f} g")
    return text
  if event.kind == detector.RELEASE:
    text = f"release: launch {event.v0_mps:+.1f} m/s"
    if event.v0_mps > 0.3:
      text += f", apex ~{detector.apex_height_m(event.v0_mps):.2f} m"
    return text
  if event.kind == detector.BOOM:
    fall = detector.fall_from_apex_m(event.airtime_s, event.v0_mps)
    return (f"LANDED {event.intensity:.2f}: airtime {event.airtime_s:.2f} s, "
            f"fall ~{fall:.2f} m, impact {event.peak_g:.1f} g")
  return (f"{event.kind}: airtime {event.airtime_s:.2f} s, "
          f"peak {event.peak_g:.1f} g")


def imu_array(rows: Sequence[Row]) -> np.ndarray:
  """Stacks samples as (n, 6) with gyro when every row has it, else (n, 3)."""
  if rows and all(r[3] is not None for r in rows):
    return np.asarray([(*r[:3], *r[3]) for r in rows], float)
  return np.asarray([r[:3] for r in rows], float).reshape(-1, 3)


def _write_imu(directory: str, name: str, imu: np.ndarray,
               header: str) -> str:
  """Writes samples in the dataset's text format.

  Args:
    directory: Destination folder, created if missing.
    name: File name with a `{stamp}` field for the local time.
    imu: (n, 3) samples in g, or (n, 6) with gyro in deg/s.
    header: Comment line written above the samples.

  Returns:
    Path of the written file.
  """
  os.makedirs(directory, exist_ok=True)
  now = time.time()
  stamp = (time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) +
           f"-{int(now * 1000) % 1000:03d}")
  path = os.path.join(directory, name.format(stamp=stamp))
  imu = np.asarray(imu, float)
  # Same precision as the stream: 0.01 g and 0.1 deg/s.
  fmt = ["%.2f"] * 3 + ["%.1f"] * (imu.shape[1] - 3)
  np.savetxt(path, imu, fmt=fmt, header=header)
  return path


def save_clip(imu: np.ndarray, event: detector.FlightEvent,
              directory: str = detector.FLIGHTS_DIR) -> str:
  """Writes a flight clip for `detector.learn`.

  Args:
    imu: (n, 3) samples in g, or (n, 6) with gyro in deg/s, pre-roll
      through post-roll.
    event: The landing event that closed the flight.
    directory: Destination folder, created if missing.

  Returns:
    Path of the written file.
  """
  header = (f"kind={event.kind} airtime_s={event.airtime_s:.3f} "
            f"v0_mps={event.v0_mps:.2f} peak_g={event.peak_g:.2f} "
            f"intensity={event.intensity:.2f}")
  return _write_imu(directory, f"flight-{{stamp}}-{event.kind}.txt", imu,
                    header)


def save_handling(imu: np.ndarray, whistles: int = 0,
                  directory: str = detector.HANDLING_DIR) -> str:
  """Writes an in-ball handling recording (no throws) for `detector.learn`.

  Args:
    imu: (n, 3) samples in g, or (n, 6) with gyro in deg/s.
    whistles: Flights the detector started during the recording.
    directory: Destination folder, created if missing.

  Returns:
    Path of the written file.
  """
  seconds = len(imu) / detector.STREAM_RATE_HZ
  header = f"kind=handling seconds={seconds:.1f} whistles={whistles}"
  return _write_imu(directory, "handling-{stamp}.txt", imu, header)


class Engine:
  """Detector plus synth plus the bookkeeping the UI and CLI need.

  `feed` runs on whatever thread delivers samples (serial reader,
  simulation thread, or offline loop) and touches the synth only through
  its thread-safe control methods. The UI drains `events` and polls
  `trace()` on its own timer.
  """

  def __init__(self, config: Optional[detector.DetectorConfig] = None,
               synth: Optional[Synth] = None, sample_rate: int = 44100,
               on_clip: Optional[ClipCallback] = None) -> None:
    self.detector = detector.FlightDetector(config or detector.load_config())
    self.synth = synth or Synth(sample_rate)
    self.events: "queue.SimpleQueue[detector.FlightEvent]" = (
        queue.SimpleQueue())
    self.on_clip = on_clip
    # Set while a simulated flight plays: live samples are dropped and the
    # flight is never saved as training data.
    self.sim_active = False
    self.samples_seen = 0
    # Samples that arrived with gyro (the MotionSynth firmware).
    self.gyro_samples = 0
    self.lifting = False
    # Thrown-flight whistles while recording handling (false triggers).
    self.handling_whistles = 0
    self._lock = threading.Lock()
    self._trace: Deque[Tuple[float, float]] = collections.deque(
        maxlen=_TRACE_LEN)
    # (t, ax, ay, az, gyro) for clips.
    self._raw: Deque[Tuple[float, float, float, float,
                           detector.Gyro]] = collections.deque(maxlen=_RAW_LEN)
    # (due time, clip start, closing event, discard).
    self._clip_due: Optional[Tuple[float, float, detector.FlightEvent,
                                   bool]] = None
    # (landing event, samples left, size scale).
    self._peak_watch: Optional[Tuple[detector.FlightEvent, int,
                                     float]] = None
    self._discard_flight = False
    self._lift_peak_st = 0.0
    self._lift_t = -math.inf
    # Pitch the thrown flight started from, and where it met the floor.
    self._offset_st = 0.0
    self._floor_st = 0.0
    # Gyro from the `GYR:` line that precedes the next `IMU:` line.
    self._gyro: detector.Gyro = None
    # Live samples while recording handling, else None.
    self._handling: Optional[List[Row]] = None

  @property
  def handling_s(self) -> Optional[float]:
    """Seconds of handling recorded so far, None when not recording."""
    rows = self._handling
    return None if rows is None else len(rows) / detector.STREAM_RATE_HZ

  def start_handling(self) -> None:
    """Starts recording in-ball handling with no throws.

    Whistles meanwhile are false triggers: they are counted in
    `handling_whistles` and never saved as flight clips.
    """
    self.handling_whistles = 0
    with self._lock:
      self._handling = []

  def stop_handling(
      self, directory: str = detector.HANDLING_DIR) -> Optional[str]:
    """Stops recording handling and saves it for `detector.learn`.

    Args:
      directory: Destination folder.

    Returns:
      Path of the saved recording, or None if under `MIN_HANDLING_S`.
    """
    with self._lock:
      rows, self._handling = self._handling, None
    if not rows or len(rows) < MIN_HANDLING_S * detector.STREAM_RATE_HZ:
      return None
    return save_handling(imu_array(rows), self.handling_whistles, directory)

  def feed_gyro(self, gx: float, gy: float, gz: float) -> None:
    """Live gyro entry point (deg/s) for the `IMU:` sample that follows."""
    self._gyro = (gx, gy, gz)

  def feed_live(self, ax: float, ay: float, az: float) -> None:
    """Live-stream entry point; ignored while a simulated flight plays."""
    gyro, self._gyro = self._gyro, None
    if not self.sim_active:
      self.feed(ax, ay, az, gyro=gyro)

  def feed(self, ax: float, ay: float, az: float, t: Optional[float] = None,
           gyro: detector.Gyro = None) -> Optional[detector.FlightEvent]:
    """Runs one sample through the detector and reacts to its event.

    Args:
      ax: X acceleration in g.
      ay: Y acceleration in g.
      az: Z acceleration in g.
      t: Sample time; defaults to `time.monotonic()` (the audio clock).
      gyro: Optional (gx, gy, gz) rotation rate in deg/s.

    Returns:
      The detector event for this sample, if any.
    """
    t = time.monotonic() if t is None else t
    event = self.detector.update(t, ax, ay, az, gyro)
    with self._lock:
      self._trace.append((t, self.detector.magnitude_g))
      self._raw.append((t, ax, ay, az, gyro))
      if self._handling is not None and not self.sim_active:
        self._handling.append((ax, ay, az, gyro))
    self.samples_seen += 1
    if gyro is not None:
      self.gyro_samples += 1
    if self.detector.state == detector.FLYING:
      self.synth.flutter_g = self.detector.flutter_g
    elif event is None:
      self._drive_lift(t)
    if self._peak_watch is not None:
      self._watch_peak()
    if event is not None:
      self._react(event)
    if self._clip_due is not None and t >= self._clip_due[0]:
      self._emit_clip()
    return event

  def _drive_lift(self, t: float) -> None:
    """Rising whoosh while the hand pushes the ball up."""
    v = self.detector.lift_mps
    if not self.synth.config.lift_whoosh:
      v = 0.0
    if not self.lifting:
      if v < LIFT_ON_MPS:
        return
      self.lifting = True
      self._lift_peak_st = 0.0
    if v < LIFT_OFF_MPS:
      self.lifting = False
      self.synth.stop_whistle(LIFT_FADE_S)
      return
    st = (LIFT_ST_PER_MPS * v + self.synth.config.semitones_per_m *
          LIFT_HEIGHT_GAIN * self.detector.lift_m)
    self._lift_peak_st = max(self._lift_peak_st, st)
    self._lift_t = t
    self.synth.lift(st)

  def _react(self, event: detector.FlightEvent) -> None:
    if event.kind == detector.RELEASE:
      self.lifting = False
      if event.bounce:
        self._hold_clip()
        self.synth.start_whistle(event.t_release, BOUNCE_V0_MPS,
                                 self._floor_st,
                                 level=BOUNCE_DECAY**event.bounce,
                                 pitch_gain=BOUNCE_PITCH_GAIN)
      else:
        carry = event.t - self._lift_t <= LIFT_CARRY_S
        self._offset_st = self._lift_peak_st if carry else 0.0
        handling = self._handling is not None and not self.sim_active
        if handling:
          self.handling_whistles += 1
        self._discard_flight = self.sim_active or handling
        self.synth.start_whistle(event.t_release, event.v0_mps,
                                 self._offset_st)
      self.events.put(event)
    elif event.kind == detector.BOOM:
      scale = BOUNCE_DECAY**event.bounce
      if not event.bounce:
        self._floor_st = self._landing_st(event)
      event = dataclasses.replace(event, intensity=scale * event.intensity)
      # Fire now; the true peak often lands one sample later (boosted).
      self.synth.stop_whistle(LANDING_FADE_S)
      self.synth.land(event.intensity)
      self._peak_watch = (event, _PEAK_WATCH_SAMPLES, scale)
    else:
      self.synth.stop_whistle(CATCH_FADE_S)
      if event.kind == detector.CAUGHT and not event.bounce:
        self.synth.land(CATCH_PUFF)
      self._landed(event)

  def _landing_st(self, event: detector.FlightEvent) -> float:
    """Whistle semitones where a thrown flight met the floor."""
    cfg = self.synth.config
    st = self._offset_st + cfg.semitones_per_m * detector.height_m(
        event.airtime_s, event.v0_mps)
    return min(max(st, -cfg.floor_st), cfg.ceiling_st)

  def _watch_peak(self) -> None:
    event, left, scale = self._peak_watch
    mag = self.detector.magnitude_g
    if mag > event.peak_g:
      size = scale * detector.boom_intensity(mag, event.airtime_s,
                                             self.detector.config.impact_g)
      event = dataclasses.replace(event, peak_g=mag, intensity=size)
      self.synth.boost(size)
    if left > 1:
      self._peak_watch = (event, left - 1, scale)
    else:
      self._peak_watch = None
      self._landed(event)

  def _landed(self, event: detector.FlightEvent) -> None:
    pending = self._clip_due
    if not event.bounce:
      # A late boom after a catch replaces the pending catch clip.
      self._clip_due = (event.t + POST_ROLL_S, event.t_release - PRE_ROLL_S,
                        event, self._discard_flight)
    elif pending is not None:
      # Bounces extend the thrown flight's clip.
      self._clip_due = (event.t + POST_ROLL_S, pending[1], pending[2],
                        pending[3])
    self.events.put(event)

  def _hold_clip(self) -> None:
    """Keeps a pending clip open until the bounce in flight ends."""
    pending = self._clip_due
    if pending is not None:
      self._clip_due = (math.inf, pending[1], pending[2], pending[3])

  def _emit_clip(self) -> None:
    _, start, event, discard = self._clip_due
    self._clip_due = None
    if self.on_clip is None or discard:
      return
    with self._lock:
      rows = [r[1:] for r in self._raw if r[0] >= start]
    if not rows:
      return
    try:
      self.on_clip(imu_array(rows), event)
    except OSError as e:
      logger.warning("Could not save flight clip: %s", e)

  def trace(self) -> Tuple[np.ndarray, np.ndarray]:
    """Snapshot of (timestamps, |a|) for plotting."""
    with self._lock:
      if not self._trace:
        return np.zeros(0), np.zeros(0)
      arr = np.asarray(self._trace)
    return arr[:, 0], arr[:, 1]

  def render_offline(self, imu: np.ndarray,
                     rate_hz: float = detector.STREAM_RATE_HZ,
                     tail_s: float = 2.5) -> np.ndarray:
    """Feeds a recording on a virtual clock and returns the mono audio.

    Args:
      imu: (n, 3) samples in g, or (n, 6) with gyro in deg/s.
      rate_hz: Sample rate of `imu`.
      tail_s: Extra seconds rendered after the last sample.

    Returns:
      Mono float32 audio at `synth.sample_rate`.
    """
    data = np.asarray(imu, dtype=float)
    has_gyro = data.ndim == 2 and data.shape[1] >= 6
    sr = self.synth.sample_rate
    dt = 1.0 / rate_hz
    blocks = []
    carry = 0.0
    t = 0.0
    for row in data:
      gyro = (row[3], row[4], row[5]) if has_gyro else None
      self.feed(row[0], row[1], row[2], t=t, gyro=gyro)
      carry += sr * dt
      n = int(carry)
      carry -= n
      blocks.append(self.synth.render(n, t))
      t += dt
    step = max(int(sr * dt), 1)
    for _ in range(int(tail_s * sr) // step):
      blocks.append(self.synth.render(step, t))
      t += step / sr
    return np.concatenate(blocks) if blocks else np.zeros(0, np.float32)


# -- Output -------------------------------------------------------------------

FALLBACK_RATE = 44100


def audio_available() -> bool:
  """Whether live speaker output can be attempted."""
  return sd is not None


def audio_unavailable_reason() -> str:
  """Why `audio_available()` is False (empty when it is True)."""
  return _AUDIO_UNAVAILABLE


def _output_device() -> Optional[Any]:
  if sd is None:
    return None
  try:
    return sd.query_devices(kind="output")
  except (sd.PortAudioError, ValueError) as e:
    logger.warning("No default audio output: %s", e)
    return None


def default_sample_rate() -> int:
  """Native rate of the default output device (44.1 kHz fallback)."""
  dev = _output_device()
  return int(dev["default_samplerate"]) if dev else FALLBACK_RATE


def output_name() -> str:
  """Name of the default output device, for status labels."""
  dev = _output_device()
  return str(dev["name"]) if dev else "no output device"


class AudioOut:
  """Streams `synth.render` blocks to the default output device."""

  def __init__(self, synth: Synth, blocksize: int = 512,
               clock: Callable[[], float] = time.monotonic) -> None:
    self.synth = synth
    self.blocksize = blocksize
    self.clock = clock
    self.underruns = 0
    self._stream: Optional[Any] = None
    self._failed = False

  @property
  def running(self) -> bool:
    return self._stream is not None and bool(self._stream.active)

  def start(self) -> None:
    """Opens and starts the stream.

    Raises:
      RuntimeError: If sounddevice or an output device is unavailable.
    """
    if self._stream is not None:
      return
    if sd is None:
      raise RuntimeError(_AUDIO_UNAVAILABLE)
    dev = _output_device()
    if dev is None:
      raise RuntimeError("no audio output device")
    channels = max(1, min(2, int(dev["max_output_channels"])))
    try:
      stream = sd.OutputStream(samplerate=self.synth.sample_rate,
                               blocksize=self.blocksize, channels=channels,
                               dtype="float32", latency="low",
                               callback=self._callback)
      stream.start()
    except sd.PortAudioError as e:
      raise RuntimeError(f"cannot open audio output: {e}") from e
    self._failed = False
    self._stream = stream
    logger.info("Audio out: %s at %d Hz, %d-frame blocks", dev["name"],
                self.synth.sample_rate, self.blocksize)

  def stop(self) -> None:
    """Stops and closes the stream (no-op when stopped)."""
    stream, self._stream = self._stream, None
    if stream is not None:
      stream.stop()
      stream.close()

  def _callback(self, outdata: np.ndarray, frames: int, unused_time: Any,
                status: Any) -> None:
    if status.output_underflow:
      self.underruns += 1
    try:
      mono = self.synth.render(frames, self.clock())
    except Exception:  # pylint: disable=broad-except
      # Keep the audio thread alive; report the first failure only.
      if not self._failed:
        logger.exception("Synth render failed; outputting silence")
        self._failed = True
      mono = np.zeros(frames, np.float32)
    outdata[:] = mono[:, None]


def write_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
  """Writes mono float samples in [-1, 1] as a 16-bit WAV file."""
  pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
  with wave.open(path, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(int(sample_rate))
    w.writeframes(pcm.tobytes())
