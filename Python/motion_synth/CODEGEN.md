# Motion Synth Codegen Plan

Local planning draft for a `motion_synth` task on [codegen-classroom](https://pypi.org/project/codegen-classroom/) (`codegen-classroom~=0.1.0`, used unmodified) that evolves Motion Synth's event detector, with side variants for landing presets and an on-device model. It mirrors [../codegen_task/README.md](../codegen_task/README.md). Anything without a link is proposed, not built.

## Starting Point

| Fact | Value | Consequence |
| :--- | :--- | :--- |
| Stream | `IMU:ax,ay,az`, 100 Hz, 0.01 g, +-16 g (15.99 g per-axis clip), no gyro yet | Lift direction and launch speed come from magnitude only |
| Positives | 12 bare-board `Drop` plus 56 reviewed foam-ball clips in `detector.FLIGHTS_DIR`, 1.6 to 2.3 s each (first 22: airtime 0.15 to 0.85 s, weightless mean 0.22 to 0.49 g, peaks 2.8 to 23 g) | A 5-sample confirmation spends a third of a 0.15 s flight |
| Negatives | 108 bare-board gestures in `detector.GESTURE_DIR` (9 classes x 12, about 1.9 min) | Too short, and not in the ball, for a per-minute false-trigger rate |
| Baseline | [`detector.learn`](detector.py)`(recordings, max_false_rate=0.07)`: 0.50 g for 5 samples (50 ms), impact 2.5 g, settle 0.75 g, refractory 0.9 s; 68/68 falls, 6/108 gesture false triggers (`Flick`, `Roll`, `Slap`). `max_false_rate=0` gives 0.55 g for 9 samples, 67/68, 0/108 | Latency versus false triggers is the core trade |
| Bias | Clips are saved only when the live detector fires, and their `kind=`/`airtime_s=` headers come from it; every kept clip has 10+ samples under 0.55 g | Recall on clips is an upper bound; headers are not ground truth |
| Review | 9 clips removed: 1 simulated, 1 dip between slams, 7 bounces with under 100 ms below 0.55 g at release | Candidate automated label-quality step |
| Cost | Reference `FlightDetector.update` takes about 1 us per sample on the host | CPU is a tiebreaker for host detectors |

## Decisions

| Question | Decision | Effect on the plan |
| :--- | :--- | :--- |
| Bounces | Each bounce gets its own short whistle and a smaller landing | Bounces are labelled flights of kind `bounce`, scored by `bounce_f1`. Today's `refractory_s` 0.9 s mutes them, so the baseline scores zero there. Bounce clips are kept as `bounce` positives instead of removed |
| Pickups | Lifting the ball off a surface whooshes | Pickups are lift positives; put-downs, carries, and swings stay negatives |
| False triggers | 1 per minute of in-ball handling is the ceiling | Promotion gate at 1/min; `false_triggers_per_min` scores `1 - v/1.0`, so lower still wins |
| Gyro | A new firmware stream adds gyro | Built as `Arduino/MotionSynth`: per 100 Hz sample it prints `GYR:gx,gy,gz` (deg/s, 1 decimal) and then the unchanged `IMU:ax,ay,az` from the same reading, plus `INFO:motion_synth gyro=1` at startup. Both hosts ([__main__.py](__main__.py) `parse_imu` and `Python/arduino_serial_handler.py`) parse `IMU:` as exactly 3 values and silently skip other lines, so appending to `IMU:` would drop every sample while a new line is ignored |

- Bounces shorter than the 50 ms confirmation can only sound at their own landing; the lever is a bounce path with a shorter confirmation inside the refractory window.
- Gyro sessions store 6 columns (`ax ay az gx gy gz`): `detector.load_xyz` keeps the first 3, so existing code still works. `Python/data` stays 3 columns because `codegen_task.data_loader.list_recordings` skips anything else.

## What To Optimize

| Track | What evolves | Why | Promotes to | Priority |
| :--- | :--- | :--- | :--- | :--- |
| Detector | Causal streaming detector: release, landing (`boom`, `caught`, late boom up to 0.6 s after a cable catch), `bounce`, lift on/off | Whistle onset latency versus false whistles; boom versus puff; bounce whistles; whoosh on pickups and upward pushes only | [calibration.json](calibration.json) or `plugins/<id>.py` | First |
| Estimators | `v0_mps`, `airtime_s`, landing peak inside the same detector | Pitch follows `height_m(t, v0)`; magnitude-only push integration turns sideways pushes and hand jitter into fake climbs | Same | With detector |
| Sound | Parameters of `synth.LANDINGS` (`boom`, `squish`, `clap`, `beep`) plus the bounce whistle | Loudness match, laptop-speaker audibility, intensity monotonicity, audio-block CPU | `synth` preset table | Second |
| On-device | TFLite Micro event model under the codegen_task deploy contract | Pays off only for an untethered ball; the tethered host has spare CPU and latency is bounded by confirmation, not compute | `model_data.h` for a firmware event mode | Optional |

Levers to seed as archetypes and refinement directions (principles, no recipes, per `config.assert_no_recipes`):

- Project acceleration onto the rest vector measured before motion to separate upward pushes from swings; with gyro, rotate that vector by the integrated rotation so "up" survives tumbling.
- With gyro, subtract the centripetal term (spin rate squared times the board's offset from the ball centre) from `|a|`; spin is one source of the 0.22 to 0.49 g weightless readings.
- Adaptive thresholds from `rest_g` and recent drag; hysteresis; a soft start that fades the whistle in after 2 to 3 samples and cancels silently if unconfirmed.
- A push-then-drop signature (above 1 g, then a sharp fall) to confirm throws sooner than quiet handling; impact-then-short-weightless for bounces.

## Data Contract

| Source | Location | Truth | Role |
| :--- | :--- | :--- | :--- |
| Gestures | `detector.GESTURE_DIR`, non-`Drop` labels | No events by construction | Negatives, `gesture_false_rate` |
| Drops | `Drop-*.txt` in the same folder | Oracle labels | Positives (bare board) |
| Clips | `detector.FLIGHTS_DIR/flight-*.txt` | Oracle labels plus review | Positives (biased) |
| Sessions | `data/sessions/*.txt` (3 or 6 columns) plus JSON sidecar | Hotkey markers refined by the oracle | Unbiased positives, negatives, lifts, bounces |
| Simulated | `detector.simulate(kind, v0_mps, drop_m, impact_g, seed)` over `detector.KINDS` | Exact | Training augmentation only, never test |

Ground truth comes from an offline, non-causal labeler, never from the live detector:

- Release onset: first sample of the first run of 8+ samples under a loose 0.65 g (look-ahead allowed).
- Landing: first sample at or above 2.0 g within `detector.LANDING_SEARCH_S` (1.5 s) is `boom`; 120 ms back in the rest band is `caught`; a spike within 0.6 s after a catch upgrades it to a late `boom`.
- Bounces: weightless runs of 3+ samples after a landing and before the ball settles are `bounce` flights with their own onset and landing.
- Lifts: labelled session takes and pickups (`lift_on`, `lift_off`); elsewhere `-1` and excluded from lift metrics.
- Label quality: automate the manual review rule as flags, not deletions: provenance check for simulated clips, dips between slams, and releases with under 100 ms below 0.55 g, which are now `bounce` candidates rather than rejects.
- Accepted labels live in `<file>.events.json` sidecars next to each recording; a per-clip PNG shows the marks for review.

Frames handed to the library:

| Frame | Rows | Columns |
| :--- | :--- | :--- |
| `X` | One per recording, or per session segment cut at rest so it holds one throw and its bounces | `xyz` (object, float32 `(n, 3)` raw g), optional `gyr` (same shape, dps, gyro sessions only), `recording_id` (unique per row), `group_id`, `source` |
| `y` | Same | `outcome` int: 0 none, 1 boom, 2 caught |
| `events` | Same | `release_idx`, `landing_idx`, `landing_kind`, `bounce_count`, `lift_on_idx`, `lift_off_idx`, `airtime_s`, `drop_height_m`; `-1` when absent; bounce times live in the sidecar table the validator reads |

- `events` reaches students via `auxiliary_train_data={"events": ...}` in `student_learning.run_parallel_classroom_learning_cycle` and `auxiliary_learning_data` in `run_parallel_classroom_final_evaluation`: the library slices it with the training fold and inserts it as a positional argument before `X_val`, so validation labels never leak. It must be NaN-free because `scoring.score_predictions` validates auxiliary frames; leave `auxiliary_test_data` unset.
- The validator gets full truth through `functools.partial` (as `deploy.deploy_validator` gets `num_classes`) and looks rows up by `recording_id`, since split indices restart at 0.
- Variable-length object columns avoid padding; the library only type-checks DataFrames (to verify nothing else assumes numeric `X`).
- Frozen test: 25 % of `group_id` values per source by stable hash (the `data_loader.assign_split` idea keyed on groups). Cycles score out of fold with `sklearn.model_selection.StratifiedGroupKFold` passed as `cv_splits` (stratify `outcome`, group `group_id`, k capped by the rarest class's group count).
- `group_id` is the session, or clips saved within 10 minutes of each other (same thrower, cable, and floor), so near-duplicates never straddle folds; a `dataset_fingerprint` over recordings and sidecars drives `--watch`.

## Student Contract

```python
def solution(X_train, y_train, events_train, X_val):
  """Fits on training rows, then runs a causal detector over each X_val stream."""
  return y_pred_df, feedback
```

- `y_pred_df`: index equal to `X_val.index`, one int column `outcome` (checked with `scoring.classification_prediction_validator`), plus:
  - `attrs["events"]`: long DataFrame `row, kind, fire_idx, est_idx, v0_mps, airtime_s`, `kind` in `release`, `boom`, `caught`, `bounce`, `lift_on`, `lift_off`; `fire_idx` is when the event became known, `est_idx` the back-dated onset or landing.
  - `attrs["detector"]`: JSON-serialisable params plus `type` (`reference` or `plugin`) for promotion.
  - `attrs["cpu_us"]`: median per-sample `update` time from `tk.run_stream`.
- The same code block defines `class Detector` with `__init__(self, params)` and `update(self, ax, ay, az, gyr=None) -> list`, so the winner exports as a plug-in and uses gyro when the stream has it.
- Causality: an event may depend only on samples up to `fire_idx`. Hidden truncated copies of some rows (same group, same fold) must reproduce the earlier events, or the validator rejects the attempt.
- `feedback`: 4 to 10 lines with hypothesis, out-of-fold release F1, latency, false triggers, and CPU.
- Environment: `numpy`, `pandas`, `scipy`, `sklearn`, `motion_synth.detector`, `synth_task.toolkit`; TensorFlow only in the on-device variant. Keep codegen_task's `final_eval_timeout_s` 480 and `cv_folds` 3 but drop `timeout_s` to about 120: a pure-Python replay pass takes under a second, so inner sweeps should use vectorized helpers in the style of `detector.max_run_below`.

Toolkit (`synth_task/toolkit.py`, mirroring [../codegen_task/toolkit.py](../codegen_task/toolkit.py)): `tk.run_stream(detector, xyz, gyr=None)` feeds samples one at a time and times `update`; `tk.reference_detector(params)` wraps `detector.FlightDetector` plus the engine's lift hysteresis (`synth.LIFT_ON_MPS` 0.8, `synth.LIFT_OFF_MPS` 0.25); `tk.features(xyz, gyr=None)` gives magnitude, jerk, rest vector, gravity-projected component, and spin-compensated magnitude; plus `tk.simulate`, `tk.match_events` (the evaluator's matcher), `tk.make_result`, and `tk.api_summary` for the prompt.

Baseline, seed, and mock student: `tk.reference_detector` with `detector.learn` thresholds fitted on the training fold, i.e. current Motion Synth. `--mock_students` runs it end to end, like `_MockStudent` returning `config.BASELINE_SOLUTION`.

## Metrics and Weights

Event metrics are registered with `metrics.register_metric(metrics.Metric(...))` and read a stash filled by `events.event_validator`, the analogue of `deploy.deploy_validator`. The library gives metric functions only flattened `y_true` and `y_pred`, calls the validator once per fold, then scores once, so unlike codegen_task's last-fold artifact stats the stash accumulates per fold: a thread-local dict keyed by `frozenset(X_val.index)`, so retries overwrite.

| Metric | Definition | Normalized score | Weight |
| :--- | :--- | :--- | ---: |
| `release_f1` | Release is a hit if it fires while the ball is truly airborne on a throw or drop | As is | 0.22 |
| `release_latency_ms` | Median `fire_idx` minus onset over hits | `1 - v/120`, floor 0 | 0.14 |
| `false_triggers_per_min` | Releases outside true flights and bounces per minute of handling or idle stream; promotion requires 1/min or less | `1 - v/1.0` | 0.12 |
| `gesture_false_rate` | Share of the 108 gestures that fire (the `max_false_rate` quantity) | `1 - v/0.1` | 0.06 |
| `landing_kind_acc` | Per true flight: boom, caught, or late boom right, fired within -20 to +80 ms | As is | 0.12 |
| `landing_timing_ms` | Median absolute landing fire error | `1 - v/50` | 0.04 |
| `bounce_f1` | Bounce whistle or landing fired within the true bounce (or at its landing when shorter than the confirmation) | As is | 0.04 |
| `lift_f1` | Whoosh on and off matched to labelled lifts and pickups; whoosh on put-downs, carries, or swings is a false positive | As is | 0.08 |
| `airtime_mae_ms` | Estimated versus oracle airtime | `1 - v/100` | 0.05 |
| `v0_consistency_mps` | `v0_mps` versus the speed implied by airtime and session `drop_height_m` | `1 - v/2` | 0.05 |
| `cpu_us_per_sample` | Median `update` time, re-measured at promotion | `1 - v/20` | 0.06 |
| `exec_time` | Library built-in, seconds per call | `inverse_normalize` | 0.02 |

- Metrics without labelled support (lifts, bounces, drop heights) are dropped from `metric_weights` for that run; `score_predictions` divides by the active weight sum, so the rest renormalize.
- `REPORT.md` adds group bootstrap intervals, per-source and per-gesture false triggers, a latency histogram, a gyro versus no-gyro split, and a timeline for each missed flight or bounce.

## Sound Preset Variant

- Prerequisite: move the constants in `synth._Hit` (`_boom`, `_squish`, `_clap`, `_beep`, `_LENGTH_S`) into a `PRESET_PARAMS` table with an optional JSON override, keeping the oscillator, RBJ low-pass, and amp graph; add the bounce whistle as one more entry.
- Rows are render conditions (`preset`, `waveform`, `intensity`): learn on intensities 0.1, 0.4, 0.7, 1.0 with `sine` and `saw`, hold out 0.25, 0.55, 0.85 with `triangle` and `square`. `outcome` is a zero carrier.
- The student returns `attrs["preset_params"]`; the validator renders each condition through `Synth.land` and `Synth.render` at 44.1 kHz with `SynthConfig.landing_volume` fixed, computes numpy/scipy descriptors, and stashes them, like `deploy.ARTIFACT_METRICS`.

| Descriptor metric | Target |
| :--- | :--- |
| Loudness spread | BS.1770-style K-weighted RMS of the first 300 ms within 3 dB across presets at equal intensity |
| Laptop audibility | At least half the energy above 200 Hz |
| Monotonicity | Spearman of intensity versus loudness and length at least 0.9 |
| Character | Per-preset bands: boom centroid falls under 300 Hz, clap shows 3 onsets about 11 ms apart, beep keeps two pitches, bounce stays under 150 ms |
| Distinctness | Minimum pairwise log-mel distance between presets above a floor |
| Hygiene | No NaN, no clicks at onset or fade, bounded pre-`tanh` peak, 4 overlapping hits render a 512-frame block in under 20 % of its 11.6 ms |

- Descriptors are proxies for taste, so promotion needs a human A/B of `build/boom_presets.wav`. A plain random or CMA search may beat LLM students while only numbers change; the classroom earns its keep when students reshape envelopes.

## On-Device Variant

- Reuse [../codegen_task/deploy.py](../codegen_task/deploy.py) pieces: `analyze_tflite`, `run_tflite`, `ALLOWED_OPS` (mirrored by `RegisterOps()` in `Arduino/Codegen`), `MAX_MODEL_KB` 64, `MAX_ARENA_KB` 40, `MIN_AGREEMENT` 0.95, `write_model_header`, and the REPLAY protocol in `device_eval_main`.
- `deploy_validator` assumes one fixed window per `X` row and one class per row (`data_loader.features_only`), so the boom validator calls the pieces directly and registers its own cost metric names.
- Contract: `attrs["tflite_model"]` maps a causal window (for example 32 x 3, or 32 x 6 with gyro) to `none`, `airborne`, `impact` scores; the evaluator slides it every sample through the reference hysteresis and requires 95 % event agreement with the returned events.
- Budget: the `deploy.py` calibration (119k MACs measured 25.9 ms, 4.8M MACs/s) leaves about 48k MACs per 10 ms sample, so per-sample inference should stay near 15k MACs (about 3 ms).
- Firmware: an event mode in `Arduino/MotionSynth` printing `EVT:` next to `GYR:` and `IMU:`; the host detector stays the fallback.

## File Layout

```text
Python/synth_task/          # mirrors Python/codegen_task
  README.md                # loop, files, scoring, commands
  config.py                # TASK_NAME = "motion_synth", DEFAULT_STUDENT_MODELS, RUN_DEFAULTS,
                           # DEFAULT_METRIC_WEIGHTS, 8-section prompt, refine options,
                           # BASELINE_SOLUTION, ARCHETYPES, insights store
  data_loader.py           # sources to X / y / events, group hash split, cv_splits, fingerprint
  labels.py                # oracle labeler, label-quality flags, sidecars, review plots
  events.py                # matcher, event_validator, fold stash, metric registration
  toolkit.py               # tk.run_stream, tk.reference_detector, tk.features, ...
  sound.py                 # preset variant: conditions, renders, descriptors
  report.py                # REPORT.md: timelines, latency, Pareto, per-source tables
  run_motion_synth.py         # runner
  run_motion_synth_test.py    # offline tests, no LLM
  leaderboard.tsv          # one row per run
  solutions/               # motion_synth_<N>s_<N>c_<date>.md, insights.json, best_report.md
Python/motion_synth/
  plugins/                 # exported winning detectors
  data/sessions/           # continuous recordings (load_recordings treats data/*.txt as flights)
Arduino/MotionSynth/          # built: GYR: then IMU: per 100 Hz sample (make build-boom)
```

- Runner flags copy [../codegen_task/run_micro_gesture.py](../codegen_task/run_micro_gesture.py): `--num_cycles`, `--num_students`, `--model_names`, `--timeout_s`, `--final_eval_timeout_s`, `--cv_folds`, `--mock_students`, `--no_seed`, `--seed_fraction`, `--respawn`, `--insights`, `--deploy`, `--deploy_margin`, `--prompt_only`, `--watch`, `--analysis`, `--report`, `--hil`, `--debug`, plus `--variant` (`detector`, `sound`, `tflm`).
- Flow reuses verified calls: `agent_config.load_env`, `reporting.setup_logger`, `get_classroom_student_configs`, the learning cycle with `cv_splits`, `auxiliary_train_data`, and `validator`, `process_cycle_results`, `scoring.get_classroom_feedback_string` plus `config.cycle_feedback`, respawn, final evaluation with `strategies=["first", "best"]`, `get_best_solution_from_eval_results`, `agents.SolutionAnalysisAgent`, `data_utils.submit_results_to_local_leaderboard`, and `reporting.build_run_report` with `write_report`.
- Runs land in `experiments/motion_synth/<experiment>/<run_id>/` (or `CODEGEN_EXPERIMENTS_DIR`). The validator pops `attrs`, so keep winners in an artifact store keyed by a prediction hash (as `deploy.stored_artifact`) or re-run the winner, which is cheap and deterministic here.

## Promotion Into Motion Synth

- Gate on the frozen test like `_promote`: only a `test_overall` gain of at least `--deploy_margin` on the same dataset fingerprint replaces the deployed record, and only at 1 false trigger per minute or fewer.
- `type == "reference"`: write [calibration.json](calibration.json) with `detector.save_calibration(config, report)`, the report carrying `solution_id`, test metrics, and fingerprint; `detector.load_config` already ignores unknown keys.
- `type == "plugin"`: export the code to `plugins/<solution_id>.py` plus params, add a `"plugin"` key to `calibration.json`, and add a `detector.PluginDetector` adapter exposing what `synth.Engine` and [widget.py](widget.py) read: `update(t, ax, ay, az) -> Optional[FlightEvent]` (gyro passed when present), `state`, `magnitude_g`, `flutter_g`, `lift_mps` and `lift_m` (mapped from `lift_on`/`lift_off`), and `config` (`impact_g` for the peak watch, thresholds for the plot). `FlightEvent.kind` gains `bounce`.
- Extra gates: the plug-in replayed by Motion Synth itself matches the solution's events on at least 95 % of test events, CPU re-measured in isolation is under budget, and `make synth-test` passes. Keep the previous calibration for rollback.

## Makefile Targets

| Target | Mirrors |
| :--- | :--- |
| `boom-classroom`, `boom-classroom-mock`, `boom-classroom-test` | `classroom`, `classroom-mock`, `classroom-test` |
| `boom-classroom-prompt`, `boom-classroom-report`, `boom-classroom-watch` | `classroom-prompt`, `classroom-report`, `classroom-watch` |
| `boom-leaderboard`, `boom-runs`, `boom-status`, `boom-report-open`, `boom-data-stats` | `leaderboard`, `runs`, `status`, `report-open`, `data-stats` (these hard-code codegen_task paths or process names) |
| `boom-label`, `boom-record`, `boom-presets` | New: labels and review plots, session recording, sound variant with `build/boom_presets.wav` |
| `build-boom`, `flash-synth`, `demo-gyro` (added) | `build-record`, `flash-record`, `record`; the port comes from `find_xiao_port`, never `$(PORT)` |
| `boom-flash-tflm`, `boom-device-eval` | `flash-codegen`, `device-eval` (on-device variant only) |

## Data Collection Plan

- Firmware: `Arduino/MotionSynth` (compiles: 97,648 B flash, 45,992 B RAM) reads `readFloatGyroX/Y/Z` from Seeed_Arduino_LSM6DS3 2.0.7 at its defaults (+-16 g, +-2000 dps, 416 Hz) and prints `GYR:` before `IMU:`. Older `IMU:`-only firmware keeps working everywhere.
- Continuous recording: `python -m motion_synth --record` and a widget toggle write every sample to `data/sessions/<stamp>.txt` (6 columns when `GYR:` lines arrive) plus a JSON sidecar (cable length, landing surface, release height, firmware, hotkey markers for throw, drop, catch, bounce, lift, pickup, handling). Nothing depends on the live detector.
- Quick win before sessions: save near-miss clips (weightless runs of 3+ samples that never confirmed) to `data/nearmiss/` to expose missed short flights and bounces.
- Positives per session: drops from 0.5, 1, and 1.5 m; vertical tosses caught at release height (clean truth, `v0 = g * airtime / 2`); throws onto carpet, couch, and hard floor (hard floors give bounces); sub-0.25 s flicks; catches by hand and by the cable going taut.
- Lifts: pickups from floor, table, and couch; slow and fast upward pushes without release; lift then throw; lift and hold. Sideways pushes must not whoosh.
- Negatives in the ball on the cable: put down, carry while walking, pendulum swing, spin, cable whip or yank, hand over, bounce in hand, shake, desk bumps.
- Targets before trusting per-minute numbers: 30 to 60 min of in-ball negatives, 100+ flights with 30+ catches, 30+ bounces, and 30+ sub-0.25 s flights, 40+ lifts including pickups, 3+ sessions on different surfaces; whole sessions go to test by group hash.
- Leave `Python/data` untouched: micro_gesture shares it and freezes its splits by hash.

## Risks and Open Questions

- Selection bias persists until sessions exist; report clip recall as an upper bound.
- Tiny data invites over-engineered detectors; grouped CV, the frozen test, the CPU term, and bootstrap intervals keep runs honest.
- Bare-board gestures differ from in-ball handling (foam damps impacts, the cable drags); treat them as a proxy.
- Self-reported `fire_idx` and CPU can be gamed; truncated copies and promotion replay cover it (cost to verify).
- Bounce whistles fight the refractory window that currently protects against double booms; the bounce path must not reopen landing false triggers.
- Pickup whooshes widen the lift trigger, so put-downs and carries need the gravity-projection or gyro lever to stay quiet.
- Library constraints (verified): metric functions see only flattened arrays; auxiliary frames become positional arguments and must be NaN-free; final evaluation with `predictions_dir` renames prediction columns to one name, so keep the single `outcome` column.
- The fold-keyed stash assumes one worker thread per student attempt, true for the library's `ThreadPoolExecutor`; pin it with a unit test.
- Live latency adds USB batching and the 512-frame audio block (11.6 ms at 44.1 kHz) to scored latency; measure once end to end.
- Port safety: the Makefile `PORT` default, `deploy.find_port`, and `report._find_port` take the first `/dev/cu.usbmodem*`, which can be the Waveshare board at `/dev/cu.usbmodem2101`; use `motion_synth.__main__.find_xiao_port` for flashing and replay, and default `--hil off`.
- `verify_task --task_dir` expects `run_<dir>.py`, so it flags `run_motion_synth.py` as it would `run_micro_gesture.py`.
- Open: bounce whistle character (short chirp versus a scaled flight whistle); whether the on-device variant is worth it without a wireless ball.

## Order of Work

1. Host parsing of `GYR:` lines (firmware done: `Arduino/MotionSynth`, flash with `make flash-synth`); `labels.py` with sidecars, flags, and review plots; near-miss saving; the session recorder.
2. Detector support for `bounce` events and pickup whooshes in Motion Synth itself, behind the same tests.
3. Task skeleton on the reference baseline: `boom-classroom-mock` green, tests for the matcher and the fold stash.
4. Real detector runs, promoting to `calibration.json` first and plug-ins once the adapter lands.
5. Sound variant after the `PRESET_PARAMS` refactor; on-device variant only if the ball goes wireless.
