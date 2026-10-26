# Teacher Notes and Classroom Findings

Reference notes distilled from `micro_gesture` classroom runs, hardware-in-the-loop (HIL) replays on the Seeed XIAO nRF52840 Sense, and ablation experiments. Specific architectural recipes and implementation details are archived here so [`config.py`](../config.py) stays concise and lets students discover improvements during classroom hill climbing.

## Physical Gesture Families and Sensor Orientation

Readings are raw accelerometer samples in `g` at `100 Hz` over `1.0 s` (`100 x 3`, `+X` toward USB-C cable edge, `+Y` toward left `D0..D6` pin edge, `+Z` up out of the chip face). Gravity (`~1.0 g` at rest) encodes board pose across four physical gesture families:

- **Baseline and Free-Fall (`Null`, `Drop`)**:
  - `Null`: flat desk rest, slightly propped cable rest, and still hand-held tremor across tilted poses (`|g| ~ 1.0 g`, dynamic amplitude `< 0.55 g`).
  - `Drop`: starts flat (`+Z ~ +1.0 g`), enters free-fall (`+Z` drops near `0 g` for `0.2-0.3 s`), and ends with a cushion impact spike.
- **Palm Strikes and Pokes (`Slap`, `Poke`)**:
  - `Slap`: grips the USB cable with ~3 inches of slack and slaps the flat board against the open palm (`10-16 g` sharp peak with cable whip).
  - `Poke`: firm no-slack grip at the sensor/cable junction (tilted `+Y ~ +0.9 g` baseline) poking the tip against the palm (`3-9 g` peak).
- **Cable Whip, Spin, and Swing (`Flick`, `Twirl`, `Pendulum`)**:
  - `Flick`: grips the cable and whips the board downward in mid-air (`-X` spike and rebound without surface impact).
  - `Twirl`: holds the cable vertically (`+X ~ +1.0 g`) and spins the board around its long axis (`Y/Z` oscillation).
  - `Pendulum`: dangles the board from the cable and swings it in a smooth low-frequency (`~1.5-2 Hz`) arc.
- **Direct Wrist Motions (`Shake`, `Punch`, `Roll`)**:
  - `Shake`: rapid lateral oscillation along `+-Y` (`3-4` cycles per second).
  - `Punch`: cable pointing up, forward thrust followed by snap-back.
  - `Roll`: `180 deg` wrist rotation flipping `Z` smoothly (`+1 g -> -1 g`) with low jerk.

## Architecture and Representation Findings

- **In-Graph Standardization vs Early Feature Bloat**:
  - Baking per-axis standardization into the graph (`tk.norm_layer(x_tr)`) is mandatory so TFLite Micro can consume raw `g` readings directly.
  - Expanding `(100, 3)` into many derived channels before the first strided convolution multiplies early activation arena, MFLOPs, and flatbuffer operator metadata (`STRIDED_SLICE`, `PAD`, `CONCATENATION` each add `~0.3-0.5 KB`).
  - A compact strided stem `Conv2D` on standardized `(100, 1, 3)` learns temporal derivatives and cross-axis differences directly in its first-layer filters at a fraction of the KB and arena cost. If explicit centered or magnitude features are added, compute them in a single minimal block or after early temporal striding.
- **Depthwise-Separable Convolutions on `(Time, 1, Channels)`**:
  - Reshaping to `(100, 1, C)` and applying a strided stem `Conv2D(..., (k, 1), strides=(2, 1), use_bias=False)` followed by `BatchNormalization` + `ReLU` and `DepthwiseConv2D((k, 1), strides=(2, 2), padding='same', use_bias=False)` + pointwise `Conv2D(..., (1, 1), use_bias=False)` keeps parameters around `1.5-2.5k`, MFLOPs around `0.04-0.07`, arena around `2.0-2.6 KB`, and flatbuffer size around `8-11 KB`.
  - When combining short (impulse) and long (swing/roll) receptive fields, sharing the stem and `1x1` pointwise projections avoids duplicating backbones.
- **Preserving Temporal Chronology and Peak Amplitude**:
  - Collapsing the window with a single `GlobalAveragePooling2D` erases the free-fall-then-impact sequence of `Drop` and the impact-vs-rebound profile of `Slap` vs `Flick`.
  - Combining `GlobalMaxPooling2D` and `GlobalAveragePooling2D`, or applying coarse 2-segment overlapping `MaxPooling2D` + `AveragePooling2D` (early half vs late half of the downsampled feature map) after a narrow `1x1 Conv2D` projection, preserves temporal ordering and peak amplitude while keeping the final `Dense` head under `1 KB`.
- **Lightweight Channel Gating**:
  - Squeeze-and-excitation gating (`MEAN` via `GlobalAveragePooling2D` + bottleneck projection + `LOGISTIC` sigmoid + `MUL`) dynamically weights gravity-pose vs high-jerk channels at near-zero MFLOPs.

## Quantization and Operator Constraints

- **Full-Integer Int8 Calibration**:
  - Representative calibration samples must span both `1 g` stillness (`Null`, `Roll`) and `16 g` palm impacts (`Slap`) so activation scales neither clip high-g spikes nor quantize low-g rotations to zero.
  - Prefer `softmax` as the final op; TFLite Micro handles `int8` softmax with a fixed output scale.
- **TensorFlow Lite Micro Operator Pitfalls**:
  - `tf.reduce_sum`, `tf.math.reduce_std`, `tf.math.reduce_variance`, and `tf.math.squared_difference` emit `SUM` and `SQUARED_DIFFERENCE`, which are not in `ALLOWED_OPS`. Use `tf.reduce_mean(..., keepdims=True)` (`MEAN`), `x * x` (`MUL`/`SQUARE`), and `tf.reduce_mean((x - m) * (x - m), ...)` (`SUB` + `MUL` + `MEAN`).
  - TensorFlow CPU `DepthwiseConv2D` requires equal row and column strides (`strides=(2, 2)` or `strides=(1, 1)`). Using `strides=(2, 1)` raises `InvalidArgumentError: Current implementation only supports equal length strides in the row and column dimensions`. On `(Time, 1, Channels)` with `padding="same"`, `strides=(2, 2)` halves `Time` while keeping width `ceil(1/2) = 1`.
  - `BatchNormalization` folds into preceding convolutions during TFLite conversion; `LayerNormalization` does not fold and emits multiple extra ops.
  - `SparseCategoricalCrossentropy` does not accept `label_smoothing` in Keras; use `tf.one_hot(y, n_classes)` with `CategoricalCrossentropy(label_smoothing=...)` and 1D `sample_weight` mapped from `tk.class_weights(y, n_classes)`.

## Augmentation and Null Boundary Hardening

- **Edge-Padded Shifts vs Circular Roll**:
  - Circular wrap-around (`np.roll`) wraps `-1 g` `Roll` tails and `Drop` impact spikes back to `t=0`, creating non-physical transitions. Edge-padded temporal shifts preserve causal onset and tail structure.
- **Pose-Preserving Augmentation and Peak-Aligned Blending**:
  - Decomposing each window into static gravity pose (`mean_t`) and dynamic impulse (`w - mean_t`) allows broader amplitude scaling on the impulse without distorting the `1.0 g` gravity vector.
  - Keeping 3D rotation jitter modest (`8-10 deg`) prevents pose-defined classes (`Roll`, `Twirl`, `Poke`, `Punch`) from rotating into one another.
  - Aligning same-class windows by their peak dynamic-energy frame before linear blending avoids artificial double-peaks on impulsive gestures (`Slap`, `Flick`, `Poke`, `Drop`).
- **Low-Energy Null Hard Negatives**:
  - Unattenuated 50% gesture cuts or chunk-shuffled high-energy windows (`Slap` at `17 g`, `Shake`) look identical to real gestures under global pooling and cause high `Null` false-positive rates (`0.30-0.45`) or collapse recall on `Roll`/`Twirl`.
  - Effective `Null` hard negatives stay below `0.55 g` peak dynamic acceleration: attenuated (`<20%`) gesture onset/tail slivers, tilted `~1 g` static holds with slow wrist drift, clamped quiet sub-windows from real recordings, and brief Hanning-windowed (`0.08-0.20 g`) desk/cable handling bumps.
