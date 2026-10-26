"""Fast offline tests for the micro_gesture task (no LLM, no TensorFlow)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from codegen_classroom import metrics as metric_lib
from codegen_classroom import scoring

from codegen_task import config
from codegen_task import data_loader
from codegen_task import deploy


@pytest.fixture(name="data_dir")
def _data_dir():
  with tempfile.TemporaryDirectory() as d:
    rng = np.random.default_rng(0)
    for label, n in (("A", 6), ("B", 5), ("Null", 4)):
      for i in range(n):
        rows = rng.integers(90, 110)
        np.savetxt(os.path.join(d, f"{label}-{i}.txt"),
                   rng.normal(size=(rows, 3)))
    yield d


def test_splits_are_grouped_and_stratified(data_dir):
  s = data_loader.load_splits(data_dir)
  assert s.class_names == ["A", "B", "Null"]
  assert s.x_train.shape[1] == data_loader.WINDOW_ROWS * 3 + 1
  ids = [set(df[data_loader.GROUP_COLUMN]) for df in
         (s.x_train, s.x_val, s.x_test)]
  assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
  assert set(s.y_test[data_loader.TARGET_COLUMN]) == {0, 1, 2}


def test_split_assignment_is_stable_when_data_grows(data_dir):
  before = data_loader.assign_split(data_loader.list_recordings(data_dir))
  np.savetxt(os.path.join(data_dir, "A-99.txt"), np.zeros((100, 3)))
  after = data_loader.assign_split(data_loader.list_recordings(data_dir))
  moved_out_of_test = [k for k, v in before.items()
                       if v == "test" and after[k] != "test"]
  assert not moved_out_of_test


def test_window_pad_and_crop():
  short = np.arange(30, dtype=np.float32).reshape(10, 3)
  long = np.arange(360, dtype=np.float32).reshape(120, 3)
  assert data_loader.window_recording(short).shape == (100, 3)
  assert data_loader.window_recording(long).shape == (100, 3)


def test_instructions_have_sections_and_no_recipes(data_dir):
  s = data_loader.load_splits(data_dir)
  txt = config.build_student_instructions(
      class_names=s.class_names, train_rows=len(s.x_train),
      val_rows=len(s.x_val), final_eval_rows=len(s.x_learning))
  config.assert_no_recipes(txt)
  for title in config.SECTION_TITLES:
    assert title in txt
  assert "attrs['tflite_model']" in txt


def test_metric_weights_registered_and_sum_to_one():
  for name in config.DEFAULT_METRIC_WEIGHTS:
    assert name in metric_lib.ALL_METRICS_CONFIG, name
  assert abs(sum(config.DEFAULT_METRIC_WEIGHTS.values()) - 1.0) < 1e-9


def test_label_metrics():
  deploy.configure(["A", "B", "Null"])
  y_true = np.array([0, 0, 1, 1, 2, 2])
  y_pred = np.array([0, 1, 1, 1, 2, 0])
  assert deploy.worst_class_recall(y_true, y_pred) == pytest.approx(0.5)
  assert deploy.null_fpr(y_true, y_pred) == pytest.approx(0.5)


def test_validator_rejects_missing_artifact():
  x_val = pd.DataFrame(np.zeros((2, 300)), columns=data_loader.FEATURE_COLUMNS)
  x_val[data_loader.GROUP_COLUMN] = ["r1", "r2"]
  y_pred = pd.DataFrame({data_loader.TARGET_COLUMN: [0, 1]}, index=x_val.index)
  with pytest.raises(scoring.ValidatorError, match="tflite_model"):
    deploy.deploy_validator([None, None, x_val], {}, (y_pred, "fb"))


def test_write_model_header(tmp_path):
  path = tmp_path / "model_data.h"
  deploy.write_model_header(str(path), bytes(range(20)), ["A", "Null"])
  text = path.read_text()
  assert "#define WINDOW_ROWS 100" in text
  assert "#define NUM_CLASSES 2" in text
  assert "g_model_len = 20" in text


def test_knowledge_scrub_and_prompt_section(tmp_path, monkeypatch):
  knowledge = config
  monkeypatch.setattr(knowledge, "INSIGHTS_PATH", str(tmp_path / "i.json"))
  assert knowledge.prompt_section("abc") == ""
  text = knowledge.scrub("use epochs=40 and dropout=0.3 with lr=1e-3")
  config.assert_no_recipes(text)

  class _R:
    def __init__(self, c, sc, m, plan):
      self.cycle, self.overall_score, self.metrics, self.plan = c, sc, m, plan

  class _S:
    cycle_results = [_R(0, 0.7, {"f1_macro": 0.8, "model_kb": 6.0,
                                 "mflops": 0.04}, "depthwise conv")]

  summary = {"artifact": {"model_kb": 6.0, "mflops": 0.04, "arena_kb": 1.0,
                          "ops": ["CONV_2D"]},
             "test_metrics": {"f1_macro": 0.9}, "test_overall_score": 0.8,
             "retrain_delta": 0.01}
  analysis = {"model_description": "tiny conv", "key_aspects": "int8",
              "next_experiments": "Try time-stretch augmentation on folds."
                                  " Add label smoothing for calibration."}
  knowledge.record_run("r1", summary, analysis, {"s": _S()}, "abc")
  sec = knowledge.prompt_section("abc")
  assert "What Previous Runs Learned" in sec and "r1" in sec
  config.assert_no_recipes(sec)
  assert len(knowledge.learned_levers()) == 2
  fb = knowledge.cycle_feedback({"s": _S()}, [])
  assert "Run frontier" in fb and "label smoothing" in fb
