# Arduino-IDE-free workflow for the XIAO nRF52840 Sense playground.
#
# Usage:
#   make setup            one-time: venv + Seeed core + libraries
#   make record           build + flash Arduino/Record, then run the UI
#   make inference        build + flash Arduino/Inference, then run the UI
#   make ui               run the PyQt playground only
#   make monitor          tail raw serial output from the board
#   make classroom        evolve the gesture model (Python/codegen_task)
#   make classroom-mock   offline smoke test of the classroom pipeline
#   make classroom-watch  rerun the classroom whenever new recordings land
#   make classroom-report rebuild REPORT.md (+HIL=on for hardware in the loop)
#   make logs             follow the latest classroom run (signal lines only)
#   make logs-raw         follow the latest classroom run, unfiltered
#   make status           running classroom process + last events
#   make leaderboard      pretty-print leaderboard.tsv
#   make runs             list experiment runs with their exports
#   make report-open      open the latest solution report
#   make data-stats       recordings per class and split sizes
#   make ports            list serial ports and who holds them
#   make codegen          build + flash Arduino/Codegen (classroom model), then run UI
#   make flash-codegen    build + flash Arduino/Codegen only
#   make device-eval      replay frozen test windows on the board
#   make demo             Motion Synth: motion-driven synth window (live board)
#   make demo-sim         Motion Synth without the board: simulated flights out loud
#   make demo-wav         render simulated flights to build/motion_synth_demo.wav
#   make synth-learn      relearn Motion Synth thresholds from Python/data + flights
#   make synth-test       offline Motion Synth tests
#   make build-synth      compile Arduino/MotionSynth (IMU: plus GYR: lines)
#   make flash-synth      build + flash Arduino/MotionSynth to the XIAO (port by USB VID)
#   make demo-gyro        flash Arduino/MotionSynth, then run the Motion Synth window

SHELL := /bin/bash
ROOT  := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
VENV  := $(ROOT)/venv
PY    := $(VENV)/bin/python
CLI   := arduino-cli --config-file $(ROOT)/arduino-cli.yaml
# Use the mbed-enabled core: the bundled TFLM fork targets the Nano 33 BLE
# style core (RingBufferN, ARDUINO_SEEED_XIAO_NRF52840_SENSE) and fails to
# build on Seeeduino:nrf52.
FQBN  := Seeeduino:mbed:xiaonRF52840Sense
PORT  ?= $(firstword $(wildcard /dev/cu.usbmodem*))
BAUD  := 115200

# The Seeed core's post-build step shells out to a bare `python`; the venv
# supplies one. Also pin Seeed's LSM6DS3 library because the official
# Arduino_LSM6DS3 in ~/Documents/Arduino/libraries shares the header name.
SEEED_IMU := $(HOME)/Documents/Arduino/libraries/Seeed_Arduino_LSM6DS3
COMPILE   := PATH="$(VENV)/bin:$$PATH" $(CLI) compile --fqbn $(FQBN) \
             --library $(SEEED_IMU)
UPLOAD    := PATH="$(VENV)/bin:$$PATH" $(CLI) upload --fqbn $(FQBN) -p $(PORT)

.PHONY: help setup venv core build-record build-inference flash-record \
        flash-inference record inference ui monitor clean \
        classroom classroom-mock classroom-watch classroom-test classroom-prompt classroom-report \
        logs logs-raw status leaderboard runs report-open data-stats ports \
        build-codegen flash-codegen codegen build-micro flash-micro micro device-eval \
        demo demo-sim demo-wav synth-learn synth-test build-synth flash-synth demo-gyro

help:
	@grep -E '^#   make' $(MAKEFILE_LIST) | sed 's/^#   //'

setup: venv core

venv:
	test -x $(PY) || python3.11 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

core:
	$(CLI) config init --overwrite --dest-file $(ROOT)/arduino-cli.yaml
	$(CLI) config set board_manager.additional_urls \
	  https://files.seeedstudio.com/arduino/package_seeeduino_boards_index.json
	$(CLI) config set library.enable_unsafe_install true
	$(CLI) core update-index
	$(CLI) core install Seeeduino:mbed
	$(CLI) lib install "Seeed Arduino LSM6DS3"
	$(CLI) lib install --zip-path $(ROOT)/tflite-micro-arduino-examples-main.zip

build-record:
	$(COMPILE) --output-dir build/Record Arduino/Record

build-inference:
	$(COMPILE) --output-dir build/Inference Arduino/Inference

flash-record: build-record
	$(UPLOAD) --input-dir build/Record Arduino/Record

flash-inference: build-inference
	$(UPLOAD) --input-dir build/Inference Arduino/Inference

record: flash-record ui

inference: flash-inference ui

ui:
	cd Python && $(PY) main.py

monitor:
	$(PY) -m serial.tools.miniterm $(PORT) $(BAUD)

# --- codegen-classroom task: Python/codegen_task -------------------------
CLASSROOM_ARGS ?=

classroom:
	cd Python && $(PY) -m codegen_task.run_micro_gesture --deploy $(CLASSROOM_ARGS)

classroom-mock:
	cd Python && $(PY) -m codegen_task.run_micro_gesture --mock_students \
	  --num_cycles=1 $(CLASSROOM_ARGS)

classroom-test:
	cd Python && $(PY) -m pytest -q codegen_task

classroom-prompt:
	cd Python && $(PY) -m codegen_task.run_micro_gesture --prompt_only

# Rebuild REPORT.md for the latest run; HIL=on flashes and replays on the board.
HIL ?= auto
classroom-report:
	cd Python && $(PY) -m codegen_task.report --run_dir latest --hil $(HIL)

# Polls Python/data; when the dataset fingerprint changes, runs a short
# seeded classroom cycle and promotes the winner if it beats the deployed one.
classroom-watch:
	cd Python && $(PY) -m codegen_task.run_micro_gesture --watch $(CLASSROOM_ARGS)

build-codegen:
	$(COMPILE) --output-dir build/Codegen Arduino/Codegen

flash-codegen: build-codegen
	$(UPLOAD) --input-dir build/Codegen Arduino/Codegen

codegen: flash-codegen ui

build-micro: build-codegen
flash-micro: flash-codegen
micro: codegen

device-eval:
	cd Python && $(PY) -m codegen_task.deploy --port $(PORT) \
	  --json $(EXP_DIR)/device_eval/device_eval_latest.json

# --- Motion Synth: Python/motion_synth (any firmware that streams IMU: lines) --
SYNTH_ARGS ?=

demo:
	cd Python && $(PY) -m motion_synth $(SYNTH_ARGS)

demo-sim:
	cd Python && $(PY) -m motion_synth --sim all $(SYNTH_ARGS)

demo-wav:
	mkdir -p build
	cd Python && $(PY) -m motion_synth --sim all --wav $(ROOT)/build/motion_synth_demo.wav $(SYNTH_ARGS)

synth-learn:
	cd Python && $(PY) -m motion_synth --learn

synth-test:
	cd Python && QT_QPA_PLATFORM=offscreen $(PY) -m pytest -q motion_synth

# Arduino/MotionSynth streams GYR:gx,gy,gz before each unchanged IMU: line.
build-synth:
	$(COMPILE) --output-dir build/MotionSynth Arduino/MotionSynth

# The port comes from find_xiao_port (USB VID, blocked ports skipped), never
# the generic $(PORT), which can grab another board's /dev/cu.usbmodem2101.
flash-synth: build-synth
	@port="$$(cd Python && $(PY) -c 'from motion_synth.__main__ import find_xiao_port as f; print(f() or "")')"; \
	if [ -z "$$port" ]; then \
	  echo "flash-synth: no XIAO nRF52840 Sense found (USB VID 0x2886); plug it in and retry" >&2; \
	  exit 1; \
	fi; \
	echo "flash-synth: uploading Arduino/MotionSynth to $$port"; \
	PATH="$(VENV)/bin:$$PATH" $(CLI) upload --fqbn $(FQBN) -p "$$port" \
	  --input-dir build/MotionSynth Arduino/MotionSynth

demo-gyro: flash-synth demo

clean:
	rm -rf build

# --- helpers -----------------------------------------------------------------
EXP_DIR   ?= $(ROOT)/experiments
LATEST_LOG = $(shell ls -t $(EXP_DIR)/*.log 2>/dev/null | head -1)
SIGNAL     = "=== Cycle|New best|Excluding|Best on test|Promoted|Keeping|Retrain|Analysis written|Report:|Hardware|Export:|deploy ok|Deploy contract|Error during"

logs:
	@test -n "$(LATEST_LOG)" || (echo "no logs in $(EXP_DIR)"; exit 1)
	@echo "following $(LATEST_LOG)"
	@tail -n 200 -f "$(LATEST_LOG)" | grep --line-buffered -E $(SIGNAL) | cut -c1-200

logs-raw:
	@test -n "$(LATEST_LOG)" || (echo "no logs in $(EXP_DIR)"; exit 1)
	@tail -n 100 -f "$(LATEST_LOG)"

status:
	@pgrep -fl "codegen_task.run_micro_gesture" || echo "no classroom process running"
	@test -n "$(LATEST_LOG)" && echo "--- $(LATEST_LOG)" && grep -E $(SIGNAL) "$(LATEST_LOG)" | tail -8 | cut -c1-160 || true

leaderboard:
	@test -f Python/codegen_task/leaderboard.tsv || (echo "no leaderboard yet"; exit 1)
	@cut -f1,2,6,7,8,9,10,13,15,21 Python/codegen_task/leaderboard.tsv | column -t -s$$'\t'

runs:
	@for d in $$(ls -d $(EXP_DIR)/micro_gesture/*/*/ 2>/dev/null); do \
	  if [ -f $$d/export/summary.json ]; then \
	    printf "%-60s score=%s\n" "$${d#$(EXP_DIR)/micro_gesture/}" \
	      "$$($(PY) -c 'import json,sys;print(round(json.load(open(sys.argv[1]))["test_overall_score"],4))' $$d/export/summary.json)"; \
	  else printf "%-60s (no export)\n" "$${d#$(EXP_DIR)/micro_gesture/}"; fi; done

report-open:
	@f=Python/codegen_task/solutions/best_report.md; test -f $$f && open $$f || echo "no report yet"

data-stats:
	@cd Python && $(PY) -c "from codegen_task import data_loader as d; import logging; logging.basicConfig(level=logging.INFO, format='%(message)s'); s=d.load_splits(); print('fingerprint', s.fingerprint)"

ports:
	@ls /dev/cu.usbmodem* 2>/dev/null || echo "no usbmodem ports"
	@for p in /dev/cu.usbmodem*; do lsof -t $$p 2>/dev/null | xargs -I{} ps -o pid=,comm= -p {} | sed "s#^#  $$p held by #"; done; true
