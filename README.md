# On-Device ML Playground

This project is a tool for beginners to explore on-device ML.

> **Upstream Credit:** Forked from [**`anandghanw/ml-on-device-playground`**](https://github.com/anandghanw/ml-on-device-playground) created by **Anandghan** (`@anandghanw`).

![screenshot](UI-Screenshot.png)

## Overview

This project provides a complete environment for recording sensor data, training machine learning models, and deploying them to microcontroller devices. It's designed to help beginners understand the full workflow of on-device machine learning without requiring extensive background knowledge.

## Codegen Classroom Extension

This fork adds an automated TinyML model discovery and hardware-in-the-loop (HIL) deployment pipeline powered by [**`codegen-classroom`**](https://github.com/google-research/codegen-classroom):

- **Evolutionary TinyML Classroom (`Python/codegen_task/`)**: Multi-agent LLM students iteratively design, quantize (`INT8`), and validate neural networks against strict microcontroller flash, arena RAM, and op-resolver budgets.
- **Classroom UI Tab (`Python/classroom_widget.py`)**: Run and monitor evolutionary cycles, view live cross-validation metrics, inspect leaderboard solutions, and trigger on-device replay directly from the PyQt GUI.
- **TFLite Micro Firmware (`Arduino/Codegen/`)**: Streams live 100 Hz IMU predictions and supports deterministic `REPLAY` verification over USB serial (`100%` host/device agreement on the 12-class gesture benchmark).
- **CLI Workflow (`Makefile`)**: Build, flash, and evaluate from the terminal:
  ```bash
  make setup          # Create virtualenv and install dependencies
  make ui             # Launch the PyQt Playground + Classroom GUI
  make classroom      # Run evolutionary model search and promote winner
  make codegen        # Compile and flash Arduino/Codegen to XIAO BLE Sense
  make device-eval    # Replay frozen test windows on-device over USB serial
  ```

See [**`Python/codegen_task/README.md`**](Python/codegen_task/README.md) for the full architecture, scoring contract, and 10-class gesture collection guide, and [**`Python/codegen_task/solutions/best_report.md`**](Python/codegen_task/solutions/best_report.md) for the latest benchmark report.

## Motion Synth Extension

[**`Python/motion_synth/`**](Python/motion_synth/README.md) turns the board into an IMU motion-driven subtractive synthesizer. Move, lift, toss, or drop the board on a USB-C cable: picking it up raises a whoosh, while it is weightless the host plays a subtractive-synth whistle (`sine` by default, plus `triangle`, `square`, `saw`, `pulse`, `organ`, `fm`, and `wind`) whose pitch follows ballistic height, the landing triggers an impact sound (`boom`, `squish`, `squelch`, `screech`, `zap`, `boing`, `shatter`, `clap`, or `beep`), and each bounce gets a smaller hop. Thresholds are learned from the recorded `Drop` class and saved flights versus every other gesture and recorded handling. It works with any firmware that streams `IMU:` lines; the optional [`Arduino/MotionSynth`](Arduino/MotionSynth/MotionSynth.ino) sketch adds `GYR:` lines for steadier lift detection.

```bash
make demo        # Motion Synth window with the live board (also a tab in `make ui`)
make demo-sim    # simulated flights, bounces, and hand moves without the board
make demo-gyro   # flash Arduino/MotionSynth (adds the gyro), then run the window
```

## Features

- Simple data collection interface
- Integrated model training pipeline
- Automated `codegen-classroom` TinyML model search and on-device HIL verification
- Real-time visualization and inference

## Table of Contents

- [Codegen Classroom Extension](#codegen-classroom-extension)
- [Motion Synth Extension](#motion-synth-extension)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
    - [Python Environment Setup](#python-environment-setup)
    - [Arduino IDE Setup](#arduino-ide-setup)
- [Getting Started](#getting-started)
    - [Data Recording](#data-recording)
    - [Model Training](#model-training)
    - [Inference](#inference)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- [XIAO BLE Sense](https://www.seeedstudio.com/Seeed-XIAO-BLE-Sense-nRF52840-p-5253.html) board
- Arduino IDE (version 1.8.x or newer) or `arduino-cli`
- Python 3.10 or newer
- USB cable for connecting the board to your computer

## Installation

### Clone this repository:
   ```bash
   git clone https://github.com/jake-g/ml-on-device-playground.git
   cd ml-on-device-playground
   ```
   
### Python Environment Setup

1. Run the `Makefile` setup target (or `./setup_python_env.sh`):
   ```bash
   make setup
   ```

   This will create a virtual environment and install all required Python packages.

### Arduino Setup

You can install all board cores and libraries automatically via `arduino-cli`:
```bash
make arduino-setup
```

Or set up manually in the Arduino IDE:

1. Install the Arduino IDE from [arduino.cc](https://www.arduino.cc/en/software)

2. Add Seeed XIAO nRF52840 Sense board support:
    - Open Arduino IDE
    - Go to `File > Preferences`
    - Add this URL to "Additional Boards Manager URLs":
      ```
      https://files.seeedstudio.com/arduino/package_seeeduino_boards_index.json
      ```
    - Go to `Tools > Board > Boards Manager...`
    - Search for "Seeed nRF52 Boards" and install

3. Install the required libraries:
    - Open the Arduino IDE
    - Go to `Sketch > Include Library > Manage Libraries...`
    - Search for and install "Seeed Arduino LSM6DS3"

4. Install the TensorFlow Lite Micro Arduino Examples library:
    - In the Arduino IDE, go to `Sketch > Include Library > Add .ZIP Library...`
    - Select the `tflite-micro-arduino-examples-main.zip` file from the root directory
    - This library has been downloaded from https://github.com/lakshanthad/tflite-micro-arduino-examples

## Getting Started

### Data Recording

1. Connect your XIAO BLE Sense board to your computer

2. Upload the recording sketch (`make record` or upload `Arduino/Record/Record.ino` in Arduino IDE)

3. Launch the playground UI:
   ```bash
   make ui
   ```

4. Follow the UI to collect training data for your gestures

### Model Training

After collecting sufficient data:

1. In the playground tool, select the **Train Model** tab for quick MLP training, or the **Classroom** tab to evolve quantized TinyML architectures with `codegen-classroom`

2. Once complete, the model will be converted to TensorFlow Lite format and prepared for deployment

### Inference

1. Upload the inference sketch (`make inference` for the Playground MLP, or `make codegen` for the Classroom model)

   **Important:** You must re-upload the sketch every time you train a new model to update the model on the device.

2. Open the **Inference** tab in the playground tool (`make ui`) to visualize real-time predictions from your model

## Project Structure

```
ml-on-device-playground/
├── Arduino/
│   ├── Record/              # 100 Hz IMU data recording sketch
│   │   └── Record.ino
│   ├── MotionSynth/         # Record plus GYR: gyro lines for Motion Synth
│   │   └── MotionSynth.ino
│   ├── Inference/           # Playground MLP inference sketch
│   │   └── Inference.ino
│   └── Codegen/             # Codegen Classroom TFLite Micro & HIL replay sketch
│       ├── Codegen.ino
│       └── model_data.h
├── Python/
│   ├── motion_synth/        # IMU motion-driven synth (detector, synth, Qt tab)
│   ├── codegen_task/        # Evolutionary TinyML task, toolkit, deploy, and reports
│   ├── data/                # Recorded 10-class 100 Hz gesture dataset
│   ├── models/              # Trained Playground models
│   ├── classroom_widget.py  # PyQt Classroom tab (live CV graph, log, report viewer)
│   └── main.py              # Main PyQt application entry point
├── Makefile                 # CLI targets for setup, UI, classroom, build, and flash
├── setup_python_env.sh      # Python environment setup script
└── tflite-micro-arduino-examples-main.zip  # TFLite Micro library
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.