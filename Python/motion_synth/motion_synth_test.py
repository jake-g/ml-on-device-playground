"""Offline tests for Motion Synth (no board, no speakers)."""

from __future__ import annotations

import glob
import math
import os

import numpy as np
import pytest

from motion_synth import __main__ as cli
from motion_synth import detector
from motion_synth import synth

_HAVE_DROPS = bool(glob.glob(os.path.join(detector.GESTURE_DIR, "Drop-*.txt")))
_needs_data = pytest.mark.skipif(not _HAVE_DROPS,
                                 reason="no Drop recordings in Python/data")


def _events(xyz, config=None):
  det = detector.FlightDetector(config or detector.DetectorConfig())
  out = []
  for i, (ax, ay, az) in enumerate(xyz):
    event = det.update(i / detector.STREAM_RATE_HZ, ax, ay, az)
    if event is not None:
      out.append(event)
  return out


def _gestures():
  return detector.load_recordings(detector.GESTURE_DIR,
                                  flights_dir="/nonexistent")


def _drain(eng):
  out = []
  while not eng.events.empty():
    out.append(eng.events.get())
  return out


def _engine(config=None, sample_rate=16000, **synth_kwargs):
  cfg = synth.SynthConfig(**synth_kwargs)
  return synth.Engine(config or detector.DetectorConfig(),
                      synth.Synth(sample_rate, cfg, seed=0))


def _rms(x):
  return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


# -- detector -----------------------------------------------------------------


@_needs_data
def test_recorded_drops_whistle_then_land():
  drops = [r for r in _gestures() if r.label == "Drop"]
  hits = sum([e.kind for e in _events(r.xyz)][:2] == ["release", "boom"]
             for r in drops)
  assert hits >= len(drops) - 1


@_needs_data
def test_strict_thresholds_never_whistle_on_gestures():
  config, report = detector.learn(_gestures(), max_false_rate=0.0)
  assert report["false_triggers"] == 0
  for rec in _gestures():
    if rec.label != "Drop":
      kinds = [e.kind for e in _events(rec.xyz, config)]
      assert detector.RELEASE not in kinds, rec.name


@pytest.mark.parametrize("kind", ["drop", "lob", "throw"])
def test_simulated_flight_estimates_physics(kind):
  v0, drop = detector.PRESETS[kind]
  events = _events(detector.simulate(kind, seed=3))
  assert [e.kind for e in events] == ["release", "boom"]
  release, landing = events
  assert release.v0_mps == pytest.approx(v0, abs=0.3)
  assert landing.airtime_s == pytest.approx(detector.airtime_s(v0, drop),
                                            abs=0.05)
  assert 0.25 <= landing.intensity <= 1.0


def test_simulated_catch_lands_softly():
  kinds = [e.kind for e in _events(detector.simulate("catch", seed=3))]
  assert kinds == ["release", "caught"]


def test_floor_hit_after_cable_catch_still_lands():
  rest = [[0.0, 0.0, 1.0]] * 50
  fall = [[0.0, 0.0, 0.2]] * 20
  taut = [[0.0, 0.0, 1.0]] * 15
  hit = [[0.0, 0.0, 9.0]]
  events = _events(np.array(rest + fall + taut + hit + rest))
  assert [e.kind for e in events] == ["release", "caught", "boom"]
  assert events[-1].peak_g == pytest.approx(9.0)


def test_bounce_after_landing_gets_its_own_flight():
  rest = [[0.0, 0.0, 1.0]] * 50
  fall = [[0.0, 0.0, 0.2]] * 20
  hit = [[0.0, 0.0, 8.0]]
  hop = [[0.0, 0.0, 1.0]] * 5 + [[0.0, 0.0, 0.2]] * 15
  events = _events(np.array(rest + fall + hit + hop + [[0.0, 0.0, 4.0]] +
                            rest))
  assert [(e.kind, e.bounce) for e in events] == [
      ("release", 0), ("boom", 0), ("release", 1), ("boom", 1)]
  assert events[2].v0_mps == 0.0


def test_impact_ringdown_is_not_a_bounce():
  rest = [[0.0, 0.0, 1.0]] * 50
  fall = [[0.0, 0.0, 0.2]] * 20
  ring = [[0.0, 0.0, 8.0], [0.0, 0.0, 0.3], [0.0, 0.0, 0.3],
          [0.0, 0.0, 0.3], [0.0, 0.0, 2.0]]
  xyz = np.array(rest + fall + ring + rest)
  assert [e.kind for e in _events(xyz)] == ["release", "boom"]


def test_simulated_bounce_chain():
  events = _events(detector.simulate("bounce", seed=1))
  assert [(e.kind, e.bounce) for e in events][:4] == [
      ("release", 0), ("boom", 0), ("release", 1), ("boom", 1)]
  assert max(e.bounce for e in events) == 2


@pytest.mark.parametrize("kind", ["putdown", "carry"])
def test_hand_motions_without_a_lift_stay_quiet(kind):
  det = detector.FlightDetector()
  peak = 0.0
  for i, (ax, ay, az) in enumerate(detector.simulate(kind, seed=1)):
    assert det.update(i / 100.0, ax, ay, az) is None
    peak = max(peak, det.lift_mps)
  assert peak < 0.5 * 0.8  # Well under the whoosh threshold.


def _turn(gyro: bool):
  """Rest, then a 90 degree turn about x in 0.5 s, then rest."""
  det = detector.FlightDetector()
  w = 180.0  # deg/s
  low = 2.0
  for i in range(150):
    angle = math.radians(w) * min(max(i - 50, 0), 50) / 100.0
    turning = 50 <= i < 100
    det.update(i / 100.0, 0.0, math.sin(angle), math.cos(angle),
               (w if turning else 0.0, 0.0, 0.0) if gyro else None)
    low = min(low, det.vertical_g)
    assert det.lift_mps < 0.8
  return low


def test_gyro_keeps_up_through_rotation():
  assert _turn(gyro=True) > 0.97
  assert _turn(gyro=False) < 0.95  # Accelerometer-only "up" lags.


def test_lift_raises_lift_speed_without_a_flight():
  det = detector.FlightDetector()
  peak = 0.0
  for i, (ax, ay, az) in enumerate(detector.simulate("lift", seed=1)):
    assert det.update(i / 100.0, ax, ay, az) is None
    peak = max(peak, det.lift_mps)
  assert 1.0 < peak < 2.0
  # Leaks back once the hand stops (down speed slower, the put-down guard).
  assert abs(det.lift_mps) < 0.3


def test_max_run_below():
  mags = np.array([1.0, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 1.0])
  assert detector.max_run_below(mags, 0.5) == 3
  assert detector.max_run_below(mags, 0.05) == 0


@_needs_data
def test_learn_budget_trades_false_triggers_for_latency(tmp_path):
  path = str(tmp_path / "calibration.json")
  config, report = detector.learn_and_save(
      flights_dir=str(tmp_path / "none"), path=path)
  strict, strict_report = detector.learn(_gestures(), max_false_rate=0.0)
  assert report["false_triggers"] <= detector.MAX_FALSE_RATE * 108 + 1
  assert config.freefall_samples < strict.freefall_samples
  assert report["positive_hits"] >= strict_report["positive_hits"]
  assert config.settle_g < 1.0 < config.impact_g
  assert detector.load_config(path) == config


def test_load_config_falls_back_to_defaults(tmp_path):
  missing = str(tmp_path / "missing.json")
  assert detector.load_config(missing) == detector.DetectorConfig()


# -- synth --------------------------------------------------------------------


def test_whistle_winds_down_and_peaks_at_apex():
  cfg = synth.SynthConfig()
  assert cfg.waveform == "sine"
  t = np.linspace(0.0, 1.0, 101)
  drop = synth.whistle_hz(t, 0.0, cfg)
  assert drop[0] == pytest.approx(cfg.pitch_hz)
  assert np.all(np.diff(drop) < 0)
  toss = synth.whistle_hz(t, 4.0, cfg)
  assert t[np.argmax(toss)] == pytest.approx(4.0 / detector.G, abs=0.02)
  lifted = synth.whistle_hz(t, 0.0, cfg, offset_st=6.0)
  assert lifted[0] == pytest.approx(cfg.pitch_hz * 2.0**0.5, rel=0.03)


@pytest.mark.parametrize("waveform", synth.WAVEFORMS + ("noise",))
def test_oscillator_is_bounded(waveform):
  inc = np.full(4410, 440.0 / 44100.0)
  phase = (np.cumsum(inc) - inc) % 1.0
  out = synth.oscillator(waveform, phase, inc, np.random.default_rng(0))
  assert np.all(np.isfinite(out))
  assert np.max(np.abs(out)) <= 1.2
  assert abs(np.mean(out)) < 0.05


def test_lowpass_removes_highs():
  rng = np.random.default_rng(0)
  amp = np.ones(8000)
  dark = synth.Voice(16000).render("saw", 200.0, 300.0, 0.7, amp, rng)
  bright = synth.Voice(16000).render("saw", 200.0, 6000.0, 0.7, amp, rng)
  assert np.std(np.diff(dark)) < 0.3 * np.std(np.diff(bright))


@pytest.mark.parametrize("waveform", synth.WAVEFORMS)
def test_offline_render_whistles_then_lands(waveform):
  eng = _engine(waveform=waveform)
  pcm = eng.render_offline(detector.simulate("throw", seed=1))
  assert np.all(np.isfinite(pcm))
  assert np.max(np.abs(pcm)) <= eng.synth.config.volume + 1e-6
  events = _drain(eng)
  assert [e.kind for e in events] == ["release", "boom"]
  release, landing = events
  # The engine finds the true 12 g peak a sample after the trigger.
  assert landing.peak_g == pytest.approx(12.0, abs=0.6)
  sr = eng.synth.sample_rate
  whistle = pcm[int((release.t + 0.2) * sr):int((landing.t - 0.05) * sr)]
  blast = pcm[int(landing.t * sr):int((landing.t + 0.3) * sr)]
  assert _rms(whistle) > 0.01
  assert _rms(blast) > _rms(whistle)


@pytest.mark.parametrize("preset", synth.LANDINGS)
def test_landing_presets_render_and_end(preset):
  s = synth.Synth(16000, synth.SynthConfig(landing=preset), seed=0)
  s.land(1.0)
  blocks = [s.render(160, k * 0.01) for k in range(400)]
  pcm = np.concatenate(blocks)
  assert np.all(np.isfinite(pcm))
  assert _rms(pcm[:1600]) > 0.02
  assert _rms(pcm[-1600:]) < 1e-4  # Every preset finishes within 4 s.


def test_whistle_and_landing_volumes_are_separate():
  quiet_whistle = _engine(whistle_volume=0.0)
  pcm = quiet_whistle.render_offline(detector.simulate("drop", seed=1))
  release, landing = _drain(quiet_whistle)
  sr = quiet_whistle.synth.sample_rate
  assert _rms(pcm[int(release.t * sr):int(landing.t * sr)]) < 1e-6
  assert _rms(pcm[int(landing.t * sr):]) > 0.01
  quiet_landing = _engine(landing_volume=0.0)
  pcm = quiet_landing.render_offline(detector.simulate("drop", seed=1))
  release, landing = _drain(quiet_landing)
  assert _rms(pcm[int(release.t * sr):int(landing.t * sr)]) > 0.01
  assert _rms(pcm[int((landing.t + 0.1) * sr):]) < 1e-6


def test_lift_plays_a_rising_whoosh():
  eng = _engine()
  hz = []
  for i, (ax, ay, az) in enumerate(detector.simulate("lift", seed=1)):
    eng.feed(ax, ay, az, t=i / 100.0)
    eng.synth.render(160, i / 100.0)
    hz.append(eng.synth.current_hz)
  hz = np.array(hz)
  sounding = np.flatnonzero(hz > 0)
  assert sounding.size > 10
  assert hz.max() > 1.5 * eng.synth.config.pitch_hz
  assert not _drain(eng)  # No flight, no landing.
  off = _engine(lift_whoosh=False)
  for i, (ax, ay, az) in enumerate(detector.simulate("lift", seed=1)):
    off.feed(ax, ay, az, t=i / 100.0)
    assert not off.lifting


def test_engine_saves_real_clips_only():
  clips = []

  def keep(xyz, e):
    clips.append((xyz, e))

  rest = np.tile([0.0, 0.0, 1.0], (100, 1))
  trace = np.vstack([rest, detector.simulate("drop", seed=2)])
  eng = _engine(sample_rate=8000)
  eng.on_clip = keep
  eng.render_offline(trace)
  assert len(clips) == 1
  xyz, event = clips[0]
  assert event.kind == detector.BOOM
  expected = (synth.PRE_ROLL_S + event.airtime_s + synth.POST_ROLL_S) * 100
  assert len(xyz) == pytest.approx(expected, abs=15)
  sim = _engine(sample_rate=8000)
  sim.on_clip = keep
  sim.sim_active = True
  sim.render_offline(trace)
  assert len(clips) == 1  # Simulated flights are never training data.


def test_save_clip_round_trips(tmp_path):
  xyz = detector.simulate("drop", seed=4)
  event = detector.FlightEvent(detector.BOOM, 1.0, 0.5, 0.0, 0.5, 9.0, 0.6)
  path = synth.save_clip(xyz, event, str(tmp_path))
  np.testing.assert_allclose(detector.load_xyz(path), xyz, atol=0.006)


def test_save_clip_keeps_gyro_columns(tmp_path):
  xyz = detector.simulate("drop", seed=4)
  gyr = np.random.default_rng(0).uniform(-500.0, 500.0, xyz.shape)
  event = detector.FlightEvent(detector.BOOM, 1.0, 0.5, 0.0, 0.5, 9.0, 0.6)
  path = synth.save_clip(np.hstack([xyz, gyr]), event, str(tmp_path))
  assert path.endswith("-boom.txt")
  got_xyz, got_gyr = detector.load_imu(path)
  np.testing.assert_allclose(got_xyz, xyz, atol=0.006)
  np.testing.assert_allclose(got_gyr, gyr, atol=0.06)


def test_imu_array_needs_gyro_on_every_row():
  full = [(0.0, 0.0, 1.0, (1.0, 2.0, 3.0))] * 3
  assert synth.imu_array(full).shape == (3, 6)
  assert synth.imu_array(full + [(0.0, 0.0, 1.0, None)]).shape == (4, 3)
  assert synth.imu_array([]).shape == (0, 3)


def test_feed_live_pairs_gyro_with_the_next_sample():
  eng = _engine(sample_rate=8000)
  eng.feed_gyro(10.0, 0.0, 0.0)
  eng.feed_live(0.0, 0.0, 1.0)
  eng.feed_live(0.0, 0.0, 1.0)
  assert (eng.samples_seen, eng.gyro_samples) == (2, 1)
  assert eng.detector.gyro_active


def test_bounces_land_smaller_inside_one_clip():
  clips = []
  eng = _engine(sample_rate=8000)
  eng.on_clip = lambda imu, e: clips.append((imu, e))
  eng.render_offline(detector.simulate("bounce", seed=1))
  events = _drain(eng)
  booms = [e for e in events if e.kind == detector.BOOM]
  assert [e.bounce for e in booms][:2] == [0, 1]
  assert booms[1].intensity < synth.BOUNCE_DECAY * booms[0].intensity
  assert synth.describe(booms[1]).startswith("BOUNCE 1")
  assert len(clips) == 1  # Named after the throw, through the last bounce.
  imu, event = clips[0]
  assert event.bounce == 0
  needed = events[-1].t - event.t_release + synth.PRE_ROLL_S
  assert len(imu) >= needed * detector.STREAM_RATE_HZ


def test_bounce_hop_whistles_from_the_floor_pitch():
  eng = _engine()
  hz = []
  for i, (ax, ay, az) in enumerate(detector.simulate("bounce", seed=1)):
    eng.feed(ax, ay, az, t=i / 100.0)
    eng.synth.render(160, i / 100.0)
    hz.append(eng.synth.current_hz)
  events = _drain(eng)
  hop = next(e for e in events if e.bounce == 1)
  end = next(e for e in events if e.bounce == 1 and e.kind != "release")
  during = hz[int(hop.t * 100) + 2:int(end.t * 100)]
  assert during and min(during) > 0.0
  assert max(during) < eng.synth.config.pitch_hz  # Near the floor pitch.


def test_handling_recording_counts_false_whistles(tmp_path):
  clips = []
  eng = _engine(sample_rate=8000)
  eng.on_clip = lambda imu, e: clips.append(e)
  eng.start_handling()
  rest = np.tile([0.0, 0.0, 1.0], (400, 1))
  trace = np.vstack([rest, detector.simulate("drop", seed=2), rest])
  for i, (ax, ay, az) in enumerate(trace):
    eng.feed(ax, ay, az, t=i / 100.0, gyro=(0.0, 0.0, 1.5))
  assert eng.handling_s == pytest.approx(len(trace) / 100.0)
  assert eng.handling_whistles == 1
  assert not clips  # A whistle while handling is a false trigger.
  path = eng.stop_handling(str(tmp_path))
  xyz, gyr = detector.load_imu(path)
  assert xyz.shape == gyr.shape == (len(trace), 3)
  assert eng.handling_s is None
  eng.start_handling()
  assert eng.stop_handling(str(tmp_path)) is None  # Too short to keep.


@_needs_data
def test_learn_holds_handling_under_ceiling(tmp_path):
  rng = np.random.default_rng(0)
  handling = np.tile([0.0, 0.0, 1.0], (6000, 1))
  handling += rng.normal(0.0, 0.02, handling.shape)
  for start in range(100, 6000, 300):  # 20 short dips in a minute.
    handling[start:start + 7] = [0.0, 0.0, 0.25]
  os.makedirs(tmp_path / "handling")
  np.savetxt(tmp_path / "handling" / "handling-test.txt", handling,
             fmt="%.2f")
  free, _ = detector.learn(
      detector.load_recordings(flights_dir=str(tmp_path / "none")))
  config, report = detector.learn(
      detector.load_recordings(flights_dir=str(tmp_path)))
  assert report["handling_minutes"] == pytest.approx(1.0)
  assert report["handling_per_min"] <= detector.MAX_HANDLING_PER_MIN
  assert config.freefall_samples > 7 >= free.freefall_samples


# -- cli ----------------------------------------------------------------------


def test_parse_imu():
  assert cli.parse_imu("IMU:0.01,-0.98,0.12") == (0.01, -0.98, 0.12)
  assert cli.parse_imu("PROB:0.1,0.9,") is None
  assert cli.parse_imu("IMU:garbage") is None


def test_parse_gyr():
  assert cli.parse_gyr("GYR:-12.3,0.4,251.7") == (-12.3, 0.4, 251.7)
  assert cli.parse_gyr("IMU:0.01,-0.98,0.12") is None
  assert cli.parse_imu("GYR:1,2,3") is None


def test_port_finder_skips_other_boards(monkeypatch):

  class Port:

    def __init__(self, device, vid, description):
      self.device, self.vid, self.description = device, vid, description

  ports = [Port("/dev/cu.usbmodem2101", 0x303A, "Waveshare"),
           Port("/dev/cu.usbmodem1101", 0x2886, "Seeed XIAO nRF52840 Sense")]
  monkeypatch.setattr(cli.list_ports, "comports", lambda: ports)
  assert cli.find_xiao_port() == "/dev/cu.usbmodem1101"
  monkeypatch.setattr(cli.list_ports, "comports", lambda: ports[:1])
  assert cli.find_xiao_port() is None


def test_cli_renders_wav(tmp_path):
  path = str(tmp_path / "demo.wav")
  assert cli.main(["--sim", "drop", "--landing", "squish", "--wav",
                   path]) == 0
  assert os.path.getsize(path) > 44100


def test_cli_replays_gyro_clip(tmp_path):
  xyz = detector.simulate("drop", seed=5)
  clip = str(tmp_path / "clip.txt")
  np.savetxt(clip, np.hstack([xyz, np.zeros_like(xyz)]), fmt="%.2f")
  wav = str(tmp_path / "clip.wav")
  assert cli.main(["--replay", clip, "--wav", wav]) == 0
  assert os.path.getsize(wav) > 44100


# -- widget -------------------------------------------------------------------


def test_widget_builds_offscreen():
  os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
  pytest.importorskip("PyQt5")
  # Lazy so the offscreen platform is set before Qt loads.
  # pylint: disable=import-outside-toplevel
  from PyQt5.QtWidgets import QApplication
  from motion_synth import widget
  # pylint: enable=import-outside-toplevel
  # pylint: disable=protected-access
  app = QApplication.instance() or QApplication([])
  tab = widget.MotionSynthWidget()
  assert tab.wave_combo.currentText() == "sine"
  tab.wave_combo.setCurrentText("fm")
  tab.landing_combo.setCurrentText("squelch")
  tab.pitch_spin.setValue(700)
  tab.whistle_slider.setValue(25)
  tab.landing_slider.setValue(75)
  tab.freefall_spin.setValue(0.45)
  cfg = tab.engine.synth.config
  assert (cfg.waveform, cfg.landing, cfg.pitch_hz) == ("fm", "squelch", 700.0)
  assert (cfg.whistle_volume, cfg.landing_volume) == (0.25, 0.75)
  assert tab.engine.detector.config.freefall_g == pytest.approx(0.45)
  tab._tick()
  tab.handling_btn.setChecked(True)
  tab.engine.feed_live(0.0, 0.0, 1.0)
  tab._tick()
  assert tab.handling_btn.text().startswith("Stop and save handling")
  assert tab.banner.text() == "HANDLING (no throws)"
  tab.handling_btn.setChecked(False)  # Under 5 s, so nothing is written.
  assert tab.engine.handling_s is None
  now = 100.0
  tab._show_event(detector.FlightEvent(detector.BOOM, now, now - 0.2, 0.0,
                                       0.2, 5.0, 0.4), now)
  assert tab.banner.text() == "SQUELCH!"
  tab._show_event(detector.FlightEvent(detector.BOOM, now, now - 0.2, 0.0,
                                       0.2, 5.0, 0.2, bounce=1), now)
  assert tab.banner.text() == "BOUNCE 1"
  tab.shutdown()
  assert app is not None
