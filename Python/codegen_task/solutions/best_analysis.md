# Winning Solution Analysis

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
