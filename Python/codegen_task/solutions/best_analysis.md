# Winning Solution Analysis

- **Winner**: `student_1_gemini_4_argon` (Cycle 5) | **Run**: `20261025_190227` | **Dataset**: `78beb5ebc3c6`
- **Tags**: `strided-conv2d, hybrid-temporal-pooling, in-graph-norm, int8, peak-aligned-blend, post-training-bias-calibration`

The model ingests raw 100x3 accelerometer windows into an in-graph normalization layer before reshaping to a 4D spatial tensor. It processes signals through two strided 2D conv blocks (5x1 kernels, stride 2x1) that downsample temporal frames from 100 to 25 across 16 channels. An overlapping 2-segment hybrid pooling layer combines AveragePooling2D and MaxPooling2D to capture dynamic peaks and energy trends, feeding a 64-dimensional representation into a compact 9-class Dense head fully quantized to int8.

## Validation Deltas (Cycle 0 → Best Cycle)

| overall_score | f1_macro | balanced_accuracy | worst_class_recall | null_fpr |
| :--- | :--- | :--- | :--- | :--- |
| 0.358 → 0.819 (+0.461) | 0.416 → 0.901 (+0.485) | 0.441 → 0.899 (+0.457) | 0.0476 → 0.741 (+0.693) | 0.445 → 0.084 (-0.361) |

| model_kb | mflops | arena_kb | params_k | mcu_ms_est | host_latency_ms | int8_fraction | exec_time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 42.5 → 7.96 (-34.6) | 0.0211 → 0.0584 (+0.0373) | 5.16 → 1.93 (-3.22) | 10.3 → 1.56 (-8.75) | 2.2 → 6.08 (+3.88) | 0.0374 → 0.00751 (-0.0299) | 0 → 1 (+1) | 4.63 → 6.6 (+1.97) |

## Key Aspects

- In-graph standardization layer (tk.norm_layer) standardizes raw 3-axis accelerometer signals on-device without external runtime preprocessing.
- Two strided 2D convolutions (5x1 kernels, 2x1 strides) downsample frame length 4x from 100 to 25, keeping parameter count at 1.56k and arena memory at 1.93 KB.
- Overlapping 2-segment hybrid temporal pooling (pool size 15, stride 10) combines average and max pooling across sub-windows to retain gesture timing and peak impulse energy.
- Peak-aligned same-class window blending aligns kinetic energy peaks across training samples before mixing to improve robustness to grip and timing shifts.
- Synthetic Null hard negatives, powered class weighting (cw^1.25), and post-training Null dense bias calibration (+0.12) protect against false gesture triggers.
- Full-integer post-training quantization calibrated on representative augmented data yields 100% int8 execution fraction and perfect prediction agreement.

## Improvements from Baseline

- Validation macro F1 improved from 0.4161 to 0.9009 (and 0.9458 on unseen test) and worst-class recall increased from 0.0476 to 0.7407 due to convolutional temporal extraction, physical data augmentation, and peak-aligned blending.
- Null false trigger rate dropped significantly from 0.4454 to 0.0840 (0.0476 on test) through synthetic background negative generation, Null weight boosting, and dense layer bias adjustment.
- Model flatbuffer size decreased from 42.54 KB to 7.96 KB and weight parameters dropped from 10.31k to 1.56k by replacing the unnormalized Flatten+Dense baseline MLP with strided convolutions and compact pooling.
- Activation arena memory was reduced from 5.16 KB to 1.93 KB due to aggressive temporal striding and compact feature aggregation.
- Int8 execution fraction reached 1.0 (up from 0.0 float32 baseline) using representative dataset calibration.

## Risks and Trade-offs

- Compute cost increased from 0.0211 to 0.0584 MFLOPs and estimated Cortex-M4F latency rose from 2.20 ms to 6.08 ms due to 2D convolution ops, though remaining well within the 18 ms budget.
- Validation worst-class recall (0.7407) remains lower than test recall (0.8000), indicating mild metric sensitivity on small fold split distributions.
- The manual post-training Null bias adjustment (+0.12) is an empirical heuristic that may require retuning if class distributions shift.

## Next Experiments

- Replace standard Conv2D layers with equal-strided DepthwiseSeparable Conv2D blocks to lower compute overhead and MCU latency back toward baseline levels.
- Incorporate native in-graph kinematic feature channels (such as jerk proxy or magnitude vectors) prior to convolution to aid gesture pair separation.
- Test multi-scale 1D temporal kernels (e.g., parallel 3x1 and 7x1 conv paths) in the stem block to capture both sharp transient impacts and extended motion sweeps.
