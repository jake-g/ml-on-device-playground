# Micro-Gesture Solution Report — Run `20261025_215858`

- **Dataset**: `78beb5ebc3c6` (9 classes: `Flick, Null, Pendulum, Poke, Punch, Roll, Shake, Slap, Twirl`)
- **Generated**: 2026-10-25T22:24:00 | **Exported**: `model.tflite`, `model_data.h`, `solution.py`, `summary.json`, `analysis.md`

## Scorecard (Frozen Test & Footprint)

| overall score | macro F1 | balanced acc | worst recall | Null FPR | TFLite agree | retrain delta |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.8754 | 0.972 | 0.9698 | 0.9167 | 0 | 1 | 0 |

| model KB | arena KB (est) | MFLOPs | params (k) | MCU ms (est) | host ms | int8 frac |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 8.133 | 1.47 | 0.06811 | 2.769 | 7.094 | 0.003885 | 1 |

## Winning Solution Analysis

- **Winner**: `student_4_gemini_3.5_flash_lite` (Cycle 11) | **Run**: `20261025_215858` | **Dataset**: `78beb5ebc3c6`
- **Tags**: `cnn-2d, full-int8-ptq, fused-conv-bn, fused-input-normalization, dual-avg-max-pooling, physics-aware-mixup, hard-negatives, cosine-decay, class-weighted-loss`

The model is a 3-layer 2D convolutional neural network with 12, 16, and 20 filters operating directly on 100x1x3 reshaped accelerometer windows. Raw inputs undergo standardized feature scaling and batch normalization analytically fused into the Conv2D weights and biases before dual temporal global average and max pooling feeds a compact dense classifier. The entire model is quantized to full int8 via post-training quantization using representative gesture and hard negative windows.

## Validation Deltas (Cycle 0 → Best Cycle)

| overall_score | f1_macro | balanced_accuracy | worst_class_recall | null_fpr |
| :--- | :--- | :--- | :--- | :--- |
| 0.356 → 0.82 (+0.464) | 0.416 → 0.915 (+0.498) | 0.441 → 0.909 (+0.468) | 0.0476 → 0.704 (+0.656) | 0.445 → 0 (-0.445) |

| model_kb | mflops | arena_kb | params_k | mcu_ms_est | host_latency_ms | int8_fraction | exec_time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 42.5 → 8.13 (-34.4) | 0.0211 → 0.0681 (+0.047) | 5.16 → 1.47 (-3.69) | 10.3 → 2.77 (-7.54) | 2.2 → 7.09 (+4.89) | 0.144 → 0.00476 (-0.139) | 0 → 1 (+1) | 2.02 → 18.3 (+16.3) |

## Key Aspects

- Analytically fuses both input normalization statistics (mean and std) and BatchNormalization layers into Conv2D weights and biases during export to eliminate runtime scaling ops.
- Utilizes a 2D convolution structure directly on (100, 1, 3) 4D tensors with temporal strides (3, 2, 1) to avoid TFLite shape manipulation overhead.
- Combines AveragePooling2D and MaxPooling2D across temporal feature maps to capture both transient impact peaks and overall motion progression.
- Employs physics-aware gravity and dynamic impulse decomposition with peak-aligned minority mixing to augment rare classes without device cost.
- Integrates hard negative background synthesis and class-weighted CategoricalCrossentropy with label smoothing to maximize minority class recall and eliminate false positives.

## Improvements from Baseline

- Replacing the unnormalized float32 MLP with a 3-layer temporal 2D CNN boosted macro F1 from 0.4161 to 0.9146 on validation and 0.9720 on test, while worst-class recall jumped from 0.0476 to 0.9167.
- Adding hard negative synthesis and minority class sample weighting completely eliminated false triggers on Null windows, reducing null_fpr from 0.4454 down to 0.0000.
- Applying full int8 quantization shrank the model flatbuffer size from 42.54 KB to 8.13 KB and parameter count from 10.31K to 2.77K.
- Activation arena memory was reduced from 5.16 KB to 1.47 KB, and host inference latency dropped from 0.144 ms to 0.0039 ms with 100% int8 operational coverage.

## Risks and Trade-offs

- Estimated MCU execution time increased from 2.20 ms to 7.09 ms due to higher compute complexity (0.068 MFLOPs vs 0.021 MFLOPs) required by multi-layer temporal convolutions.
- The multi-stage synthetic mixing pipeline relies heavily on static gravity decomposition assumptions that could be sensitive to extreme mounting angles or sensor noise.

## Next Experiments

- Investigate depthwise separable convolutions across temporal layers to reduce MFLOPs and estimated Cortex-M4F cycle latency.
- Explore multi-scale kernel widths in early convolutional layers to capture both high-frequency impacts and slower continuous gestures simultaneously.
- Test post-softmax probability calibration or dynamic thresholding on out-of-fold predictions to further optimize worst-class recall.

## Dataset & Physical Separability

Fingerprint `78beb5ebc3c6` | **178** recordings (96 train / 37 val / 45 frozen test) | `100x3` window @ 100 Hz | raw length 97–113 rows (median 107).
- **Pose-separable gestures**: `Roll` flips `Z` gravity negative, `Twirl` and `Punch` hold the cable vertical (`+X ~ +0.8g, Z ~ 0g`), and `Poke` maintains a tilted firm-grip baseline (`+Y ~ +0.75g`).
- **Impulse & rhythm-separable gestures**: `Slap` reaches `~15g` peak dynamic impact with high jerk, `Flick` snaps and rebounds along the cable, `Shake` oscillates at `~4-8 Hz`, `Pendulum` swings smoothly, and `Null` stays below `~0.5g` dynamic amplitude.

| class | train | val | test | total | mean pose (X,Y,Z) g | peak a_dyn (g) | RMS jerk g/s | dom Hz |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Flick | 10 | 4 | 5 | 19 | (-2.18, +0.33, +0.17) | 12.52 | 146.0 | 1.5 |
| Null | 12 | 5 | 6 | 23 | (+0.09, +0.01, +0.26) | 0.18 | 5.4 | 5.7 |
| Pendulum | 10 | 4 | 5 | 19 | (-1.84, -0.14, +0.04) | 2.64 | 22.1 | 3.5 |
| Poke | 13 | 5 | 6 | 24 | (-0.08, +0.65, +0.33) | 5.68 | 102.9 | 1.8 |
| Punch | 10 | 4 | 5 | 19 | (+0.14, -0.06, +0.45) | 9.95 | 135.3 | 1.0 |
| Roll | 10 | 4 | 4 | 18 | (-0.13, -0.17, +0.38) | 2.83 | 52.1 | 1.7 |
| Shake | 10 | 4 | 5 | 19 | (-1.91, +0.40, +0.43) | 15.12 | 304.3 | 8.9 |
| Slap | 11 | 4 | 5 | 20 | (-1.00, +0.09, +0.22) | 16.12 | 274.1 | 1.1 |
| Twirl | 10 | 3 | 4 | 17 | (-3.97, +0.27, +0.57) | 3.57 | 59.8 | 4.8 |
| total | 96 | 37 | 45 | 178 | - | - | - | - |

![dataset_hist](plots/micro_gesture_4s_12c_20261025_v2_dataset_hist.png)

*Figure 1. Recording length and acceleration magnitude distributions (1 g mode at rest).*

![kinematic_space](plots/micro_gesture_4s_12c_20261025_v2_kinematic_space.png)

*Figure 2. (a) Gravity pose vs. peak dynamic impulse across all 120 recordings; (b) per-class physical feature matrix explaining why raw pose + centered + jerk channels separate the 10 classes.*

![class_signatures](plots/micro_gesture_4s_12c_20261025_v2_class_signatures.png)

*Figure 3. Per-class mean ± 1 SD acceleration over the 1 s window.*

## Model Architecture, Internals & Budgets

- **I/O & Quantization**: input `[1, 100, 3] int8 scale=0.13173 zp=6` → output `[1, 9] int8 scale=0.0039062 zp=-128` | int8=True (100% int8 ops) | **34,053** MACs, **2,964** weight bytes
- **Operators (7)**: `AVERAGE_POOL_2D, CONCATENATION, CONV_2D, FULLY_CONNECTED, MAX_POOL_2D, RESHAPE, SOFTMAX`

| metric | model KB | arena KB (est) | MFLOPs | params (k) | MCU ms (est) | host ms |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| value | 8.133 | 1.47 | 0.06811 | 2.769 | 7.094 | 0.003885 |
| soft / hard budget | 16.0 / 64.0 | 6.0 / 40.0 | 0.2 / - | 6.0 / - | 18.0 / - | 0.05 / - |

### Layer Breakdown

| # | op | output | dtype | MACs | weight B | live act B |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | RESHAPE | 1x100x1x3 | int8 | 300 | 16 | 600 |
| 1 | CONV_2D | 1x32x1x12 | int8 | 8064 | 300 | 684 |
| 2 | CONV_2D | 1x14x1x16 | int8 | 13440 | 1024 | 608 |
| 3 | CONV_2D | 1x12x1x20 | int8 | 11520 | 1040 | 464 |
| 4 | AVERAGE_POOL_2D | 1x1x1x20 | int8 | 20 | 0 | 260 |
| 5 | MAX_POOL_2D | 1x2x1x20 | int8 | 40 | 0 | 280 |
| 6 | CONCATENATION | 1x3x1x20 | int8 | 60 | 0 | 120 |
| 7 | RESHAPE | 1x60 | int8 | 60 | 8 | 120 |
| 8 | FULLY_CONNECTED | 1x9 | int8 | 540 | 576 | 69 |
| 9 | SOFTMAX | 1x9 | int8 | 9 | 0 | 18 |

![macs_per_op](plots/micro_gesture_4s_12c_20261025_v2_macs_per_op.png)

*Figure 4. Per-op (a) compute (kMACs), (b) weight footprint (KB), and (c) live activation RAM pressure (KB) in execution order.*

![model_internals](plots/micro_gesture_4s_12c_20261025_v2_model_internals.png)

*Figure 5. (a) t-SNE projection of the quantized 60D penultimate feature representation across all 120 recordings (stars = frozen test); (b) sliding 200 ms temporal occlusion sensitivity on the true-class logit margin.*

## TFLite Evaluation (Validation and Frozen Test)

| split | rows | accuracy | macro F1 | mean top-1 prob (all / ok / err) |
| :--- | :--- | :--- | :--- | :--- |
| val | 131 | 1 | 1 | 0.957 (ok 0.957, err nan) |
| test | 159 | 0.9748 | 0.972 | 0.919 (ok 0.926, err 0.651) |

| class | val F1 (support) | test precision | test recall | test F1 (support) |
| :--- | :--- | :--- | :--- | :--- |
| Flick | 1 (n=12) | 1 | 0.9333 | 0.966 (n=15) |
| Null | 1 (n=35) | 0.9767 | 1 | 0.988 (n=42) |
| Pendulum | 1 (n=12) | 1 | 1 | 1 (n=15) |
| Poke | 1 (n=15) | 0.9444 | 0.9444 | 0.944 (n=18) |
| Punch | 1 (n=12) | 0.9375 | 1 | 0.968 (n=15) |
| Roll | 1 (n=12) | 0.9167 | 0.9167 | 0.917 (n=12) |
| Shake | 1 (n=12) | 1 | 0.9333 | 0.966 (n=15) |
| Slap | 1 (n=12) | 1 | 1 | 1 (n=15) |
| Twirl | 1 (n=9) | 1 | 1 | 1 (n=12) |

![confusion](plots/micro_gesture_4s_12c_20261025_v2_confusion.png)

*Figure 6. Row-normalized confusion matrices on validation (a) and frozen test (b) splits; cell labels are counts.*

## Classifier Calibration & ROC/PR (Held-Out Val + Test, 290 rows)

| accuracy | macro F1 | top-2 acc | macro AUC | macro AP | MCC | kappa | log loss | ECE (10 bins) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.9862 | 0.9851 | 0.9966 | 0.9998 | 0.9984 | 0.984 | 0.984 | 0.09009 | 0.04805 |

| class | support | precision | recall | f1 | ROC AUC | AP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Flick | 27 | 1 | 0.963 | 0.9811 | 0.9999 | 0.9987 |
| Null | 77 | 0.9872 | 1 | 0.9935 | 0.9997 | 0.9992 |
| Pendulum | 27 | 1 | 1 | 1 | 1 | 1 |
| Poke | 33 | 0.9697 | 0.9697 | 0.9697 | 0.9998 | 0.9983 |
| Punch | 27 | 0.9643 | 1 | 0.9818 | 0.9999 | 0.9987 |
| Roll | 24 | 0.9583 | 0.9583 | 0.9583 | 0.9997 | 0.9966 |
| Shake | 27 | 1 | 0.963 | 0.9811 | 0.9993 | 0.9942 |
| Slap | 27 | 1 | 1 | 1 | 1 | 1 |
| Twirl | 21 | 1 | 1 | 1 | 1 | 1 |

![curves](plots/micro_gesture_4s_12c_20261025_v2_curves.png)

*Figure 7. (a) One-vs-rest ROC, (b) precision-recall, (c) calibration reliability diagram, (d) top-1 confidence.*

![per_class](plots/micro_gesture_4s_12c_20261025_v2_per_class.png)

*Figure 8. Per-class precision, recall, and F1 on all held-out rows.*

## Streaming Simulation (50% Overlap)

| windows | duration (s) | pure-window acc | Null FPR | boundary windows | prediction flips |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 89 | 45.0 | 0.96 | 0.1429 | 39 | 48 |

![stream](plots/micro_gesture_4s_12c_20261025_v2_stream.png)

*Figure 9. Sliding 1 s windows (50% overlap) over concatenated test recordings: (a) predicted vs true class, (b) top-1 confidence vs UNSURE threshold.*

## Host Benchmark

`Python 3.11.17` | `TF 2.15.1` | `macOS-26.7.1-arm64-arm-64bit` | Evaluator e2e: **0.015 ms/row**.

| runtime | p50 ms | p95 ms | min ms | inf/s |
| :--- | :--- | :--- | :--- | :--- |
| TFLite reference (no delegate) | 0.004459 | 0.00571 | 0.004375 | 2.243e+05 |
| TFLite + XNNPACK | 0.001375 | 0.001375 | 0.001291 | 7.273e+05 |

![host_latency](plots/micro_gesture_4s_12c_20261025_v2_host_latency.png)

*Figure 10. Single-threaded host latency over 300 invocations (reference kernel vs XNNPACK).*

## Hardware in the Loop (XIAO nRF52840 Sense)

Flashed `Arduino/Codegen` on `/dev/cu.usbmodem2101` (44 s) | Flash `257208 B`, Static RAM `98280 B` | `INFO:model_bytes=8328,window_rows=100,classes=Flick|Null|Pendulum|Poke|Punch|Roll|Shake|Slap|Twirl,input_type=int8,output_type=int8,step=50`.

| host acc | device acc | host F1 | device F1 | device latency ms | arena bytes | host-device agreement |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 1 | 1 | 1 | 4.94 (max 4.94) | 2788 | 1 |

## Pareto Frontier & Hardware Trade-offs

**242** scored attempts across **7** runs on dataset `78beb5ebc3c6` (non-dominated accuracy-vs-size frontier):

| KB | val F1 | MFLOPs | arena KB | val score | run | student | cycle |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 5.2 | 0.713 | 0.046 | 1.7 | 0.6886 | 20261025_170637 | 1_gemini_4_argon | 1 |
| 6.4 | 0.889 | 0.058 | 1.9 | 0.8184 | 20261025_170637 | 1_gemini_4_argon | 3 |
| 6.4 | 0.905 | 0.058 | 1.9 | 0.8227 | 20261025_170637 | 1_gemini_4_argon | 4 |
| 7.7 | 0.916 | 0.086 | 1.8 | 0.8185 | 20261025_215858 | 2_gemini_3.5_flash | 10 |
| 8.1 | 0.918 | 0.068 | 1.5 | 0.8232 | 20261025_215858 | 1_gemini_4_argon | 10 |
| 8.1 | 0.919 | 0.068 | 1.5 | 0.8203 | 20261025_215858 | 1_gemini_4_argon | 6 |
| 8.1 | 0.923 | 0.068 | 1.5 | 0.8304 | 20261025_215858 | 3_gemini_3.1_pro_preview | 11 |
| 8.9 | 0.923 | 0.050 | 1.7 | 0.8255 | 20261025_212043 | 1_gemini_4_argon | 9 |
| 8.9 | 0.926 | 0.050 | 1.7 | 0.8323 | 20261025_212043 | 1_gemini_4_argon | 8 |
| 8.9 | 0.927 | 0.050 | 1.7 | 0.8302 | 20261025_212043 | 1_gemini_4_argon | 10 |
| 8.9 | 0.927 | 0.050 | 1.7 | 0.8223 | 20261025_212043 | 3_gemini_3.1_pro_preview | 10 |
| 11.2 | 0.931 | 0.059 | 2.3 | 0.8183 | 20261025_190227 | 3_gemini_3.1_pro_preview | 6 |
| 13.8 | 0.933 | 0.110 | 3.2 | 0.7696 | 20261024_190925 | 3_gemini_3.1_pro_preview | 7 |

![pareto](plots/micro_gesture_4s_12c_20261025_v2_pareto.png)

*Figure 11. Multi-objective Pareto dashboard: (a) validation F1 vs model KB, (b) validation F1 vs MFLOPs, (c) activation arena RAM vs model flash KB colored by F1, and (d) overall score vs model KB showing the 8–14.5 KB sweet spot.*

## Student LLM Elo Ratings & Head-to-Head Record

Pairwise smooth-margin Elo (`start=1500`, `K_cycle=12`, `K_test=24`) across **7** runs (**68** classroom rounds) on dataset `78beb5ebc3c6`:

| LLM family | Elo | win rate (matches) | cycle #1s | test #1s | best val | best test | mean KB |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| gemini-4-argon | 1634 | 73.1% (193) | 38 | 5 | 0.8323 | 0.8611 | 12.8 |
| gemini-3.5-flash | 1529 | 59.2% (193) | 13 | 0 | 0.8242 | 0.8650 | 14.3 |
| gemini-3.1-pro-preview | 1493 | 42.5% (193) | 9 | 1 | 0.8304 | 0.8588 | 16.2 |
| gemini-3.5-flash-lite | 1343 | 22.0% (171) | 1 | 1 | 0.8203 | 0.8754 | 15.4 |

![elo_ratings](plots/micro_gesture_4s_12c_20261025_v2_elo_ratings.png)

*Figure 12. (a) LLM student Elo trajectory across all classroom cycles and frozen-test evaluations; (b) pairwise head-to-head win rate matrix.*

## Classroom Progression

| student | best cycle | val score | val F1 | KB | MFLOPs | scored cycles |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1_gemini_4_argon | 10 | 0.8232 | 0.9176 | 8.133 | 0.06811 | 12 |
| 2_gemini_3.5_flash | 9 | 0.8202 | 0.9191 | 8.133 | 0.06811 | 12 |
| 3_gemini_3.1_pro_preview | 11 | 0.8304 | 0.9225 | 8.133 | 0.06811 | 12 |
| 4_gemini_3.5_flash_lite | 11 | 0.8203 | 0.9146 | 8.133 | 0.06811 | 12 |

![progress](plots/micro_gesture_4s_12c_20261025_v2_progress.png)

*Figure 13. Per-student learning trajectory across cycles (lines = best-so-far, diamonds = peer collaboration).*

![final_test](plots/micro_gesture_4s_12c_20261025_v2_final_test.png)

*Figure 14. Frozen-test metrics per student.*

## Leaderboard History

| student | test_overall | elo | test_f1_macro | test_wcr | model_kb | mflops | arena_kb | retrain_delta | solution_id | dataset | timestamp |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| student_1_gemini_4_argon_r1 | 0.819 | 1400 | 0.924 | 0.833 | 10.79 | 0.067 | 2.58 | -0.002 | micro_gesture_4s_8c_20261023_v3 | 78beb5ebc3c6 | 2026-10-23T02:38:21 |
| student_4_gemini_3.5_flash_lite | 0.816 | 1390 | 0.933 | 0.786 | 10.95 | 0.059 | 2.58 | -0 | micro_gesture_4s_8c_20261023_v4 | 78beb5ebc3c6 | 2026-10-23T04:10:11 |
| student_2_gemini_3.5_flash | 0.786 | 1325 | 0.886 | 0.75 | 11.09 | 0.049 | 3.22 | 0 | micro_gesture_4s_8c_20261024 | 78beb5ebc3c6 | 2026-10-24T15:56:19 |
| student_1_gemini_4_argon | 0.842 | 1497 | 0.942 | 0.778 | 9.45 | 0.048 | 1.93 | 0 | micro_gesture_4s_8c_20261024_v2 | 78beb5ebc3c6 | 2026-10-24T19:27:54 |
| student_1_gemini_4_argon | 0.855 | 1587 | 0.942 | 0.75 | 6.43 | 0.058 | 1.93 | 0 | micro_gesture_4s_8c_20261024_v3 | 78beb5ebc3c6 | 2026-10-24T19:55:09 |
| student_1_gemini_4_argon | 0.861 | 1629 | 0.944 | 0.833 | 6.42 | 0.058 | 1.93 | 0 | micro_gesture_4s_5c_20261025 | 78beb5ebc3c6 | 2026-10-25T17:31:38 |
| student_1_gemini_4_argon_r2 | 0.849 | 1543 | 0.95 | 0.833 | 8.21 | 0.069 | 2.15 | 0 | micro_gesture_4s_8c_20261025 | 78beb5ebc3c6 | 2026-10-25T18:05:20 |
| student_1_gemini_4_argon | 0.855 | 1587 | 0.946 | 0.8 | 7.96 | 0.058 | 1.93 | 0 | micro_gesture_4s_8c_20261025_v2 | 78beb5ebc3c6 | 2026-10-25T19:17:01 |
| student_3_gemini_3.1_pro_preview | 0.852 | 1565 | 0.95 | 0.778 | 8.91 | 0.05 | 1.68 | 0 | micro_gesture_4s_12c_20261025 | 78beb5ebc3c6 | 2026-10-25T21:49:07 |
| student_4_gemini_3.5_flash_lite | 0.875 | 1701 | 0.972 | 0.917 | 8.13 | 0.068 | 1.47 | 0 | micro_gesture_4s_12c_20261025_v2 | 78beb5ebc3c6 | 2026-10-25T22:23:16 |
