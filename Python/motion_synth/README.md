# Motion Synth

An IMU motion-driven synthesizer for the XIAO nRF52840 Sense: move, lift, toss, or drop the board on a USB-C cable. Pick it up and a whistle rises; let it fly and the pitch winds down as it falls; land it and a `boom`, `squish`, `squelch`, `screech`, `zap`, `boing`, `shatter`, `clap`, or `beep` goes off, and every bounce after that gets its own smaller hop. It uses plain subtractive synthesis with no samples.

## Quick Start

```bash
make demo         # window: live board, speakers (tab also lives in `make ui`)
make demo-sim     # no board: simulated flights, bounces, and hand moves out loud
make demo-wav     # offline render to build/motion_synth_demo.wav
make demo-gyro    # flash Arduino/MotionSynth (adds the gyro), then run the window
make synth-learn  # relearn thresholds from Python/data plus saved recordings
make synth-test   # offline tests
```

Any firmware that streams `IMU:ax,ay,az` at 100 Hz works (`Arduino/Record` or `Arduino/Codegen`). [`Arduino/MotionSynth`](../../Arduino/MotionSynth/MotionSynth.ino) also prints a `GYR:gx,gy,gz` line (deg/s) before each unchanged `IMU:` line, which keeps the lift detector's sense of "up" right while the board turns. `make flash-synth` finds the XIAO by USB vendor id, never the generic `PORT`, and `make flash-codegen` puts the Codegen firmware back. Run either `make demo` or the `make ui` tab, not both at once, since each opens the port. Options pass through, for example `make demo SYNTH_ARGS="--waveform fm --landing squelch --pitch 700"`.

## Sound Controls

| Control | Constant in [`synth.py`](synth.py) | CLI flag |
| :--- | :--- | :--- |
| Waveform (`sine` default): sine, triangle, square, saw, pulse, organ, fm, wind | `WAVEFORM` | `--waveform` |
| Landing sound (`boom` default): boom, squish, squelch, screech, zap, boing, shatter, clap, beep | `LANDING` | `--landing` |
| Pitch at release | `PITCH_HZ` | `--pitch` |
| Master, whistle, and landing volume | `VOLUME`, `WHISTLE_VOLUME`, `LANDING_VOLUME` | `--volume`, `--whistle_volume`, `--landing_volume` |
| Filter cutoff (x pitch) and resonance | `CUTOFF_RATIO`, `RESONANCE` | tab only |
| Rising whoosh when lifted | `LIFT_WHOOSH` | `--no_lift` |
| Bounce size, rebound speed, hop pitch arc | `BOUNCE_DECAY`, `BOUNCE_V0_MPS`, `BOUNCE_PITCH_GAIN` | constants only |

## How It Works

- **Free fall**: an accelerometer reads about 0 g whenever the board is in flight, going up or coming down. On the cable it reads 0.2 to 0.5 g because of drag and spin. The whistle starts after `freefall_samples` readings below `freefall_g`, back-dated to the moment of release.
- **Whistle pitch follows height**: one octave per metre around the pitch at release. A toss climbs to the apex, then winds down faster and faster, getting louder as the fall speeds up. The launch speed comes from integrating the reading above 1 g over the 0.4 s before release.
- **Lift**: the detector tracks which way is up (the gyro turns it with the board when the stream has `GYR:` lines; slow 1 g readings correct drift). Only the push along up counts toward the lift speed that drives the rising whoosh, and on release the flight whistle continues from that pitch. Downward speed leaks away slower than upward speed, so a put-down's final braking only cancels it: a simulated put-down peaks at 0.28 m/s and a sideways carry at 0.03 m/s, against 0.8 m/s to start the whoosh (a lift reaches 1.07). On the bare-board recordings the whoosh never fires on `Punch` or `Poke` and fires on 3 to 5 of 12 `Shake`, `Slap`, `Flick`, and `Roll` recordings. Without the gyro, swings (`Pendulum`, `Twirl`) still whoosh; the gyro stream is meant to fix that.
- **Landing**: the first reading at or above `impact_g` fires the landing sound right away. The next 3 samples are checked for the true peak, which can raise the sound's size. Size depends on peak g and airtime. A soft catch, where readings come back near 1 g (often just the cable going taut), gets a quiet version, and a floor hit up to 0.6 s later still triggers the full landing.
- **Bounces**: for `refractory_s` (0.9 s) after a landing, only bounces can start: after a 40 ms ring-down, 3 weightless samples (`bounce_samples`) are enough. Each of up to 3 bounces plays a short hop whistle, starting from the pitch where the throw met the floor with its few centimetres of height exaggerated, then a landing at `BOUNCE_DECAY` (half) the size of the one before. Replaying the 56 saved flights, 34 bounce 1 to 3 times, with landings shrinking like 0.82, 0.34, 0.11. Bounces extend the throw's saved clip instead of saving clips of their own.
- **Synth**: one subtractive `Voice` runs an oscillator (`sine` by default, or `triangle`, `square`, `saw`, `pulse`, `organ`, `fm`, `wind`, or `noise`) into a resonant RBJ low-pass and then an amp, and it plays every sound. The whistle uses the selected waveform with the filter tracking its pitch. Landing presets: `boom` (noise sweeping bright to dark plus a sub thump), `squish` (single resonant dive plus a bloop), `squelch` (bubbly high-Q acid wobble), `screech` (metallic FM glide plus skid noise), `zap` (exponential ray-gun sweep), `boing` (twangy spring pitch bounce), `shatter` (staggered noise shards plus stepped glassy FM pings), `clap` (three quick noise bursts), and `beep` (two falling tones).

## Learned From the Data and the Spec

`detector.learn` grid-searches the free-fall threshold and the confirmation length. Positives are the recorded `Drop` class plus your saved flights in [`data/`](data). Negatives are every other gesture in `Python/data`. Bare-board slaps and flicks may trigger up to `MAX_FALSE_RATE` (7%) of the time, because padded throws rarely move like that and flights on a short cable need a fast whistle. `--max_false_rate 0` gives the strict thresholds. Recorded handling may start at most `MAX_HANDLING_PER_MIN` (1) whistle per minute. The current `calibration.json`:

| Setting | Value | Evidence |
| :--- | :--- | :--- |
| `freefall_g`, `freefall_samples` | 0.50 g for 5 samples (50 ms) | Catches 68/68 real falls (all 56 saved flights, all 12 drops); strict (`--max_false_rate 0`) needs 0.55 g for 9 samples (90 ms) and catches 67/68 |
| False triggers | 6/108 gestures | Only `Flick`, `Roll`, `Slap` on the bare board |
| `bounce_samples` | 3 samples (30 ms) | Catches 7/7 clips in `data/bounces/` |
| Handling | none recorded yet | Record some so learning can hold it to 1 whistle per minute |
| `impact_g` | 2.5 g | `Null` rest and hand-held readings top out at 1.52 g |
| `settle_g` | 0.75 g | `Null` readings never go below 0.70 g |
| Flight readings | 0.01 to 0.50 g weightless (median 0.27); landings 1.5 to 24.5 g | Recorded throws on the 6 to 10 ft cable |

These numbers come from the device itself: the LSM6DS3's +-16 g full scale (impacts clip at 15.99 g per axis), the 0.01 g, 100 Hz stream, and the chip's own free-fall engine ceiling of 500 mg.

**Recording handling**: press **Record handling (no throws)** in the tab, or run `python -m motion_synth --headless --record_handling`, then carry the board around, pick it up, put it down, and swing it on the cable without letting go. Press again (or Ctrl-C) to save it to `data/handling/`. Whistles during the recording count as false triggers and are never saved as flights. Then click **Learn from recorded data** or run `make synth-learn`.

**Data hygiene**: the tab saves every real flight to `data/`, with gyro columns when the stream has them; flights from the test buttons are never saved. Review of the first 65 saved clips:
- **1 simulated drop**, written before that fix: removed.
- **1 dip between violent slams**: removed.
- **7 events with under 100 ms below 0.55 g at release**: bounces and bobbles caught by the faster detector, moved to `data/bounces/`. Learning reports how many the bounce confirmation catches but uses them as neither positives nor negatives.

## Files

| File | Contents |
| :--- | :--- |
| [`detector.py`](detector.py) | Device spec, ballistic physics, flight, bounce, and lift detector with gravity tracking, threshold learning, simulated flights and hand moves |
| [`synth.py`](synth.py) | Subtractive synth and landing presets, the engine that plays flight events and records clips and handling, speaker and WAV output |
| [`widget.py`](widget.py) | PyQt5 tab (shared serial handler in `make ui`, standalone in `make demo`) |
| [`__main__.py`](__main__.py) | CLI: window, `--headless` (with `--record_handling`), `--sim`, `--replay`, `--wav`, `--learn` |
| [`motion_synth_test.py`](motion_synth_test.py) | Offline tests on the real recordings, simulations, synth, CLI, and widget |
| `calibration.json` | Learned thresholds plus the evidence behind them |
| `data/` | Saved flight clips (`ax ay az [gx gy gz]` text, header holds the detected event) |
| `data/bounces/`, `data/handling/` | Bounce examples, and handling with no throws |
