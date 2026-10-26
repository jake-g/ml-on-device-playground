"""Headless classroom runner for the micro_gesture task.

Also hosts `--watch` (rerun on new recordings) and `--prompt_only`
(render PROMPT.md), mirroring the template's `--report_only`.

Runs a population of Gemini students that evolve `solution()` against the
deploy contract, evaluates the survivors on the frozen test recordings,
appends to `leaderboard.tsv`, exports the winning TFLite model plus a
firmware-ready `model_data.h`, and optionally promotes it to
`Arduino/Codegen/model_data.h` when it beats the deployed model.

Hill climbing across runs: unless `--no_seed` is given, half the students
start from the winning code in the top leaderboard solution report
(`solutions/micro_gesture_<N>s_<N>c_<date>.md`) so a rerun refines rather than
restarts; the rest start from distinct archetypes.

Usage:
  python -m codegen_task.run_micro_gesture
  python -m codegen_task.run_micro_gesture --mock_students
  python -m codegen_task.run_micro_gesture --num_cycles=4 --deploy
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import functools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, MutableMapping, Sequence

from absl import app
from absl import flags
from absl import logging as absl_logging
import numpy as np
import pandas as pd

from codegen_classroom import agent_config
from codegen_classroom import agents
from codegen_classroom import code_execution
from codegen_classroom import data_utils
from codegen_classroom import reporting
from codegen_classroom import scoring
from codegen_classroom import student_learning

from codegen_task import config
from codegen_task import data_loader
from codegen_task import deploy
from codegen_task import report as report_lib

FLAGS = flags.FLAGS
RUN_DEFAULTS = config.RUN_DEFAULTS

flags.DEFINE_integer("num_cycles", RUN_DEFAULTS["num_cycles"],
                     "Learning cycles.")
flags.DEFINE_integer("num_students", RUN_DEFAULTS["num_students"],
                     "Students in the classroom.")
flags.DEFINE_list("model_names", list(config.DEFAULT_STUDENT_MODELS),
                  "Student LLM model names, cycled across students.")
flags.DEFINE_integer("timeout_s", RUN_DEFAULTS["timeout_s"],
                     "Per-solution execution timeout in seconds.")
flags.DEFINE_integer("final_eval_timeout_s", RUN_DEFAULTS["final_eval_timeout_s"],
                     "Final evaluation timeout in seconds.")
flags.DEFINE_integer("patience", RUN_DEFAULTS["patience"],
                     "Cycles without improvement before mutation.")
flags.DEFINE_integer("max_mutations", RUN_DEFAULTS["max_mutations"],
                     "Mutation attempts before a student is excluded.")
flags.DEFINE_integer("max_output_tokens", RUN_DEFAULTS["max_output_tokens"],
                     "Max output tokens per completion.")
flags.DEFINE_string("experiment_name", "micro_gesture", "Experiment group.")
flags.DEFINE_string("data_dir", None, "Recordings dir (default Python/data).")
flags.DEFINE_bool("mock_students", False,
                  "Run the deterministic baseline instead of calling Gemini.")
flags.DEFINE_bool("no_seed", True,
                  "Start from the Vanilla Original baseline instead of"
                  " seeding from the previous best leaderboard solution.")
flags.DEFINE_float("seed_fraction", RUN_DEFAULTS["seed_fraction"],
                   "Fraction of students seeded when --no_seed=False; the"
                   " rest start fresh so the population keeps exploring.")
flags.DEFINE_bool("respawn", True,
                  "Replace excluded students with fresh ones seeded from the"
                  " current best so the population never shrinks.")
flags.DEFINE_bool("insights", True,
                  "Inject lessons from previous runs into the prompt and"
                  " record this run's lessons afterwards.")
flags.DEFINE_integer("cv_folds", RUN_DEFAULTS["cv_folds"],
                     "Score each cycle on grouped K-fold out-of-fold"
                     " predictions over train+val (0 = single hold-out val).")
flags.DEFINE_bool("deploy", False,
                  "Promote the winner to Arduino/Codegen/model_data.h"
                  " when it beats the currently deployed model.")
flags.DEFINE_bool("prompt_only", False,
                  "Render PROMPT.md from the current config and exit.")
flags.DEFINE_bool("watch", False,
                  "Poll the data dir and rerun on fingerprint change.")
flags.DEFINE_float("watch_interval", 30.0,
                   "Polling interval in seconds when --watch is enabled.")
flags.DEFINE_float("watch_quiet", 120.0,
                   "Quiet period in seconds without new files before triggering"
                   " a --watch classroom run.")
flags.DEFINE_bool("watch_once", False,
                  "Check once, run if changed, and exit when --watch is set.")
flags.DEFINE_string("retrain_solution", "",
                    "Re-fit a specific leaderboard solution_id (or 'best') on"
                    " the current dataset and export to Arduino/Codegen.")
flags.DEFINE_float("deploy_margin", 0.005,
                   "Minimum test overall_score gain required to redeploy.")
flags.DEFINE_bool("analysis", True,
                  "Run the solution analyzer on the winner versus the baseline.")
flags.DEFINE_bool("report", True,
                  "Build export/REPORT.md with figures and benchmarks.")
flags.DEFINE_enum("hil", "auto", ["auto", "on", "off"],
                  "Hardware-in-the-loop step: flash the winner and replay the"
                  " test split on the board if a free serial port exists.")
flags.DEFINE_bool("debug", False,
                  "Enable verbose debug logging for student attempts,"
                  " feedback, and validator.")

agent_config.load_env()

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.normpath(os.path.join(TASK_DIR, "..", ".."))
FIRMWARE_DIR = os.path.join(REPO_DIR, "Arduino", "Codegen")
SOLUTIONS_DIR = os.path.join(TASK_DIR, "solutions")
KEEP_TOP_SOLUTIONS = 10
DEPLOYED_JSON = os.path.join(FIRMWARE_DIR, "deployed.json")


class _LoggedStudentAgent(agents.StudentAgent):
  """StudentAgent that logs its extracted plan (and code on --debug) immediately."""

  def generate_solution(self, prompt: str) -> str:
    sol = super().generate_solution(prompt)
    try:
      code, plan = code_execution.extract_python_code(
          sol, target_fn_name=config.STUDENT_FUNCTION_NAME
      )
      plan_clean = (plan or "").strip()
      if not plan_clean and code:
        doc_m = re.search(r'"""(.*?)"""', code, re.DOTALL)
        if doc_m:
          plan_clean = doc_m.group(1).strip()
      if plan_clean:
        lg = logging.getLogger("codegen_classroom")
        if FLAGS.debug:
          lg.info("[%s] Plan:\n%s", self.name, plan_clean)
        else:
          lg.info("[%s] Plan: %s", self.name,
                  plan_clean.replace("\n", " ")[:320])
      if FLAGS.debug and code:
        logging.getLogger("codegen_classroom").debug(
            "[%s] Generated Code (%d chars):\n%s",
            self.name, len(code), code,
        )
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    return sol


class _MockStudent(agents.StudentAgent):
  """Returns the reference baseline without calling an LLM."""

  def generate_code(self, prompt: str) -> str:
    del prompt
    self.generate_code_count += 1
    return config.BASELINE_SOLUTION

  def generate_solution(self, prompt: str) -> str:
    return self.generate_code(prompt)


def _runs_base() -> str:
  base = os.environ.get("CODEGEN_EXPERIMENTS_DIR",
                        os.path.join(REPO_DIR, "experiments"))
  return os.path.join(base, config.TASK_NAME, FLAGS.experiment_name)


def _read_seed() -> str | None:
  """Winning code from the top leaderboard solution report or live run CSVs."""
  if FLAGS.no_seed:
    return None
  lb = os.path.join(TASK_DIR, "leaderboard.tsv")
  if os.path.isfile(lb):
    df = pd.read_csv(lb, sep="\t")
    if not df.empty and "solution_id" in df:
      df = df[df.get("tags", "").fillna("").astype(str) != "mock"]
      for _, row in df.sort_values("test_overall", ascending=False).iterrows():
        path = os.path.join(SOLUTIONS_DIR, f"{row['solution_id']}.md")
        if not os.path.isfile(path):
          continue
        with open(path, encoding="utf-8") as f:
          text = f.read()
        top = text.find("## Top Solution 1")
        start = text.find("```python", top if top >= 0 else 0)
        end = text.find("\n```", start + 9)
        if start >= 0 and end > start:
          return text[start:end + 4]
  # Fallback: best valid attempt from any active or recent run on disk.
  import glob  # pylint: disable=import-outside-toplevel
  best_score = -1.0
  best_code = None
  pattern = os.path.join(REPO_DIR, "experiments", "micro_gesture", "*", "*",
                         "students", "*.csv")
  for csv_path in sorted(glob.glob(pattern), key=os.path.getmtime,
                         reverse=True)[:12]:
    try:
      sdf = pd.read_csv(csv_path)
    except Exception:  # pylint: disable=broad-exception-caught
      continue
    if "overall_score" not in sdf or "code" not in sdf:
      continue
    for _, r in sdf.iterrows():
      sc = r.get("overall_score")
      code = r.get("code")
      if pd.notna(sc) and float(sc) > best_score and isinstance(code, str) and "def solution" in code:
        best_score = float(sc)
        best_code = code if "```python" in code else f"```python\n{code}\n```"
  return best_code


def _validate_seed(seed: str, splits: data_loader.Splits, logger) -> str | None:
  """Runs the seed once under the current budget; returns it or None."""
  code, _ = code_execution.extract_python_code(seed)
  if not code:
    return None
  fn_args = [splits.x_train, splits.y_train, splits.x_val]
  t0 = time.time()
  try:
    out, _ = code_execution.execute_code(
        code, config.STUDENT_FUNCTION_NAME, fn_args=fn_args,
        initial_namespace={}, timeout_seconds=FLAGS.timeout_s)
    functools.partial(deploy.deploy_validator,
                      num_classes=len(splits.class_names))(fn_args, {}, out)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning("Seed solution failed under the current budget (%s);"
                   " falling back to the baseline.", str(e)[:160])
    return None
  logger.info("Seed solution validated in %.0fs", time.time() - t0)
  return seed


def _materialize_best(
    code: str, splits: data_loader.Splits, run_dir: str,
    best_name: str = "", best_eval: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
  """Captures the winning TFLite artifact.

  Prefers the exact model the final evaluation scored (looked up from the
  validator's artifact store via the saved predictions); re-runs the code
  only when that lookup fails.
  """
  stats = model_bytes = None
  pred_df = None
  exec_time = float((best_eval or {}).get("metrics_exec_time", 0.0) or 0.0)
  pred_csv = os.path.join(run_dir, "predictions",
                          f"{best_name}_best_y_predictions.csv")
  if best_name and os.path.isfile(pred_csv):
    saved = pd.read_csv(pred_csv, index_col=0)
    hit = deploy.stored_artifact(
        deploy.prediction_key(saved.index, saved.iloc[:, 0].to_numpy()))
    if hit is not None:
      model_bytes, stats = hit
      pred_df = pd.DataFrame({splits.y_test_eval.columns[0]:
                              saved.iloc[:, 0].to_numpy(int)},
                             index=splits.x_test_eval.index)
      absl_logging.info("Using the final-evaluation artifact for %s",
                        best_name)
  if model_bytes is None:
    absl_logging.warning("Final-eval artifact not found; re-fitting %s",
                         best_name or "winner")
    fn_args = [splits.x_learning, splits.y_learning, splits.x_test_eval]
    out, exec_time = code_execution.execute_code(
        code, config.STUDENT_FUNCTION_NAME, fn_args=fn_args,
        initial_namespace={}, timeout_seconds=FLAGS.final_eval_timeout_s,
    )
    functools.partial(deploy.deploy_validator,
                      num_classes=len(splits.class_names))(fn_args, {}, out)
    stats = deploy.current_stats()
    model_bytes = deploy.current_model_bytes()
    pred_df = out[0]
  if stats is None or model_bytes is None:
    return None
  deploy._tls.stats = stats  # pylint: disable=protected-access
  score, metrics, _, _ = scoring.score_predictions(
      splits.y_test_eval, pred_df, {"exec_time": exec_time},
      dict(config.DEFAULT_METRIC_WEIGHTS),
  )
  export_dir = os.path.join(run_dir, "export")
  os.makedirs(export_dir, exist_ok=True)
  with open(os.path.join(export_dir, "model.tflite"), "wb") as f:
    f.write(model_bytes)
  with open(os.path.join(export_dir, "solution.py"), "w",
            encoding="utf-8") as f:
    f.write(code)
  comment = (
      f"run {os.path.basename(run_dir)} dataset {splits.fingerprint}\n"
      f"test overall_score {score:.4f} f1_macro {metrics.get('f1_macro', 0):.3f}"
      f" model {stats.model_kb:.1f} KB mflops {stats.mflops:.3f}"
      f" arena_est {stats.arena_kb:.1f} KB int8 {stats.quantized}"
  )
  deploy.write_model_header(
      os.path.join(export_dir, "model_data.h"), model_bytes,
      splits.class_names, extra_comment=comment,
  )
  from sklearn import metrics as sk_metrics  # pylint: disable=import-outside-toplevel
  y_true_arr = splits.y_test_eval.iloc[:, 0].to_numpy(int)
  y_pred_arr = pred_df.loc[splits.y_test_eval.index].iloc[:, 0].to_numpy(int)
  cm = sk_metrics.confusion_matrix(
      y_true_arr, y_pred_arr, labels=list(range(len(splits.class_names)))
  ).tolist()
  summary = {
      "run_dir": run_dir,
      "dataset_fingerprint": splits.fingerprint,
      "class_names": splits.class_names,
      "test_overall_score": score,
      "test_metrics": {k: float(v) for k, v in metrics.items()},
      "confusion_matrix": cm,
      "artifact": stats.as_dict(),
      "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
  }
  data_utils.save_dict_to_json(summary, os.path.join(export_dir, "summary.json"))
  return summary


def _solution_id(n_students: int, n_cycles: int) -> str:
  """`micro_gesture_<N>s_<N>c_<YYYYMMDD>`, suffixed _v2.. on collision."""
  base = (f"{config.TASK_NAME}_{n_students}s_{n_cycles}c_"
          f"{pd.Timestamp.now().strftime('%Y%m%d')}")
  sid, k = base, 1
  while os.path.exists(os.path.join(SOLUTIONS_DIR, f"{sid}.md")):
    k += 1
    sid = f"{base}_v{k}"
  return sid


def _compact_run_markdown(md: str, has_deployment_analysis: bool) -> str:
  """Removes redundant prompt dumps and verbose boilerplate from run reports."""
  # Replace the 350+ line verbatim Student Instructions blockquote with a link.
  md = re.sub(
      r"## Student Instructions\n\n(?:>.*\n?)+",
      "## Student Instructions\n\n"
      "*Full task prompt and deploy contract: [`PROMPT.md`](../PROMPT.md)*\n\n",
      md,
  )
  # Replace the 35-line double-spaced Objectives list with a compact summary.
  obj_summary = (
      "## Objectives\n\n"
      "- **Prediction Quality (0.68)**: `f1_macro` (0.42),"
      " `worst_class_recall` (0.11), `balanced_accuracy` (0.09),"
      " `null_fpr` (0.06, lower is better)\n"
      "- **On-Device Cost (0.22)**: `model_kb` ≤ 16 KB (0.08),"
      " `mflops` ≤ 0.20 (0.05), `arena_kb` ≤ 6 KB (0.03),"
      " `host_latency_ms` ≤ 0.05 ms (0.02), `mcu_ms_est` ≤ 18 ms (0.02),"
      " `params_k` ≤ 6k (0.02)\n"
      "- **Deploy Hygiene (0.10)**: `int8_fraction` (0.05),"
      " `tflite_agreement` (0.03), `exec_time` (0.02)\n\n"
  )
  md = re.sub(
      r"## Objectives\n\n(?:-   \*\*.*\n(?:    .*\n)*\n?)+",
      obj_summary,
      md,
  )
  # If Deployment Analysis is appended, omit the library's duplicate Analysis
  # block (which formats list items as comma-joined paragraphs).
  if has_deployment_analysis:
    md = re.sub(
        r"## Analysis\n\n.*?(?=\n## Top Solution 1:|\Z)",
        "",
        md,
        flags=re.S,
    )
  return md


def _write_solution(sol_id: str, run_dir: str, students, metric_weights,
                    row: Mapping[str, Any], analysis_md: str, logger) -> None:
  """solutions/<id>.md in the library format plus our deployment analysis."""
  sections = reporting.build_run_report(
      run_dir, students=list(students.values()), metric_weights=metric_weights,
      solution_id=sol_id, metadata=dict(row), top_n=1, worst_n=0)
  path = os.path.join(SOLUTIONS_DIR, f"{sol_id}.md")
  reporting.write_report(path, sections, precision=3)
  with open(path, encoding="utf-8") as f:
    cleaned = _compact_run_markdown(f.read(), bool(analysis_md))
  with open(path, "w", encoding="utf-8") as f:
    f.write(cleaned)
  if analysis_md:
    append_analysis(path, analysis_md)
  logger.info("Solution written: solutions/%s.md", sol_id)


def append_analysis(path: str, analysis_md: str) -> None:
  """Appends the deployment analysis as a level-2 section.

  Written after `reporting.write_report`, which rewrites every
  `![..](plots/..)` link inside a markdown section to the progress plot.
  Omit sections (`Classroom Progression`, `Leaderboard History`) that are
  already shown at the top of `solutions/<id>.md`.
  """
  body = analysis_md.split("\n", 1)[1] if analysis_md.startswith("# ") else analysis_md
  body = re.sub(
      r"\n## Classroom [Pp]rogression\n.*",
      "",
      body,
      flags=re.S,
  )
  body = re.sub(r"^(#{2,5}) ", lambda m: "#" + m.group(1) + " ", body,
                flags=re.M)  # nest under this section
  with open(path, "a", encoding="utf-8") as f:
    f.write(f"\n## Deployment Analysis\n\n{body.strip()}\n")


def _prune_solutions(keep_extra: Sequence[str] = ()) -> None:
  """Keeps the top KEEP_TOP_SOLUTIONS by test score plus `keep_extra`."""
  lb = os.path.join(TASK_DIR, "leaderboard.tsv")
  if not os.path.isfile(lb):
    return
  df = pd.read_csv(lb, sep="\t")
  if "solution_id" not in df or df.empty:
    return
  df = df[df.get("tags", "").fillna("").astype(str) != "mock"]
  keep = set(df.sort_values("test_overall", ascending=False)
             .head(KEEP_TOP_SOLUTIONS)["solution_id"].astype(str))
  keep |= set(keep_extra)
  plots = os.path.join(SOLUTIONS_DIR, "plots")
  for name in os.listdir(SOLUTIONS_DIR):
    if name.endswith(".md") and name.startswith(config.TASK_NAME + "_"):
      if name[:-3] not in keep:
        os.remove(os.path.join(SOLUTIONS_DIR, name))
  if os.path.isdir(plots):
    for name in os.listdir(plots):
      if name.endswith(".png") and not any(name.startswith(k) for k in keep):
        os.remove(os.path.join(plots, name))


def _promote(summary: dict[str, Any], run_dir: str) -> bool:
  """Copies the exported header into the firmware dir if it is better."""
  current = {}
  if os.path.isfile(DEPLOYED_JSON):
    current = data_utils.load_json_from_file(DEPLOYED_JSON)
  prev = float(current.get("test_overall_score", -1.0))
  new = float(summary["test_overall_score"])
  same_data = current.get("dataset_fingerprint") == summary["dataset_fingerprint"]
  if same_data and new < prev + FLAGS.deploy_margin:
    absl_logging.info("Keeping deployed model (%.4f >= %.4f)", prev, new)
    return False
  os.makedirs(FIRMWARE_DIR, exist_ok=True)
  shutil.copy(os.path.join(run_dir, "export", "model_data.h"),
              os.path.join(FIRMWARE_DIR, "model_data.h"))
  data_utils.save_dict_to_json(summary, DEPLOYED_JSON)
  exp_deploy_dir = os.path.join(REPO_DIR, "experiments", "deployed")
  os.makedirs(exp_deploy_dir, exist_ok=True)
  data_utils.save_dict_to_json(
      summary, os.path.join(exp_deploy_dir, "deployed_latest.json")
  )
  data_utils.save_dict_to_json(
      summary,
      os.path.join(exp_deploy_dir, f"deployed_{os.path.basename(run_dir)}.json"),
  )
  absl_logging.info("Promoted model to %s (score %.4f, previous %.4f)",
                    FIRMWARE_DIR, new, prev)
  return True


ANALYSIS_FIELDS = {
    "model_tags": "Comma-separated tags for architecture, quantization, and"
                  " preprocessing (e.g. depthwise-conv, int8, in-graph-norm).",
    "model_description": "2-3 sentences describing the final model and how it"
                         " consumes the raw 100x3 accelerometer window.",
    "key_aspects": "3-6 bullet-style sentences on the design choices that"
                   " matter most for on-device accuracy, size, and latency.",
    "improvements_from_baseline": "3-6 sentences on what changed relative to"
                                  " the baseline code and which metric deltas"
                                  " each change explains.",
    "regressions_or_risks": "1-3 sentences on anything that got worse or is"
                            " fragile (overfitting, quantization drift,"
                            " budget headroom).",
    "next_experiments": "3 concrete single-lever experiments for the next"
                        " classroom run, ordered by expected gain.",
}


def _fmt_metrics(m: Mapping[str, Any]) -> str:
  keys = ("overall_score", "f1_macro", "balanced_accuracy",
          "worst_class_recall", "null_fpr", "model_kb", "mflops", "arena_kb",
          "params_k", "int8_fraction", "host_latency_ms", "mcu_ms_est")
  parts = []
  for k in keys:
    v = m.get(k, m.get(f"metrics_{k}"))
    if v is not None:
      parts.append(f"{k}={float(v):.4g}")
  return ", ".join(parts)


def _as_code(text: str) -> str:
  """Returns the python body whether `text` is fenced or already raw."""
  if not text:
    return ""
  if "```python" in text:
    try:
      return code_execution.extract_python_code(text)[0]
    except Exception:  # pylint: disable=broad-exception-caught
      return ""
  return text


def _analyze_winner(
    students: Mapping[str, agents.StudentAgent],
    best_name: str,
    best_code: str,
    best_eval: Mapping[str, Any],
    summary: dict[str, Any],
    instructions: str,
    run_dir: str,
    logger,
) -> dict[str, Any] | None:
  """Analyzes the winning solution against the cycle-0 baseline."""
  student = students[best_name]
  base_code, base_val = "", {}
  if student.cycle_results:
    first = student.cycle_results[0]
    base_code = _as_code(first.code or "")
    base_val = dict(first.metrics or {})
    base_val["overall_score"] = first.overall_score
  if not base_code:
    base_code = _as_code(config.BASELINE_SOLUTION)
  b = student.best_cycle
  best_val = {}
  if 0 <= b < len(student.cycle_results):
    best_val = dict(student.cycle_results[b].metrics or {})
    best_val["overall_score"] = student.cycle_results[b].overall_score
  plan = student.cycle_results[b].plan if 0 <= b < len(student.cycle_results) else ""

  extra = (
      "\n\nThe baseline this run started from (cycle 0) is shown below, with"
      " its validation metrics. Compare the final solution against it when"
      " filling in improvements_from_baseline and regressions_or_risks.\n"
      f"Baseline validation metrics: {_fmt_metrics(base_val)}\n"
      f"Final validation metrics: {_fmt_metrics(best_val)}\n"
      f"Final frozen-test metrics: {_fmt_metrics(summary['test_metrics'])},"
      f" overall_score={summary['test_overall_score']:.4f}\n"
      f"Final artifact: {json.dumps(summary['artifact'])}\n\n"
      f"Baseline code:\n```python\n{base_code.strip()}\n```\n"
  )
  analyzer = agents.SolutionAnalysisAgent(
      config=agents.AgentConfig(name="solution_analysis"),
      run_base_dir=run_dir, json_fields=ANALYSIS_FIELDS,
  )
  try:
    result = analyzer.analyze(
        student_name=best_name, best_code=best_code, best_plan=plan or "",
        best_test_metrics=best_eval, best_val_metrics=best_val,
        student_instructions=instructions,
        metrics_description=scoring.get_metrics_description(
            dict(config.DEFAULT_METRIC_WEIGHTS)),
        task_description_for_prompt=config.TASK_TITLE,
        extra_prompt_instructions=extra,
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.warning("Solution analysis failed: %s", e)
    return None
  if not result:
    return None

  def _delta(k: str) -> str:
    a, c = base_val.get(k), best_val.get(k)
    if a is None or c is None:
      return "n/a"
    return f"{float(a):.3g} → {float(c):.3g} ({float(c) - float(a):+.3g})"

  raw_tags = result.get("model_tags", "")
  if isinstance(raw_tags, (list, tuple)):
    raw_tags = ", ".join(str(x) for x in raw_tags)
  desc = str(result.get("model_description", "") or "").strip()

  q_keys = ("overall_score", "f1_macro", "balanced_accuracy",
            "worst_class_recall", "null_fpr")
  c_keys = ("model_kb", "mflops", "arena_kb", "params_k", "mcu_ms_est",
            "host_latency_ms", "int8_fraction", "exec_time")

  lines = [
      "# Winning Solution Analysis", "",
      f"- **Winner**: `{best_name}` (Cycle {b}) | **Run**:"
      f" `{os.path.basename(run_dir)}` | **Dataset**:"
      f" `{summary['dataset_fingerprint']}`",
  ]
  if raw_tags:
    lines.append(f"- **Tags**: `{raw_tags}`")
  if desc:
    lines += ["", desc]
  lines += [
      "", "## Validation Deltas (Cycle 0 → Best Cycle)", "",
      "| " + " | ".join(q_keys) + " |",
      "| " + " | ".join(":---" for _ in q_keys) + " |",
      "| " + " | ".join(_delta(k) for k in q_keys) + " |",
      "",
      "| " + " | ".join(c_keys) + " |",
      "| " + " | ".join(":---" for _ in c_keys) + " |",
      "| " + " | ".join(_delta(k) for k in c_keys) + " |",
      "",
  ]
  titles = {
      "key_aspects": "Key Aspects",
      "improvements_from_baseline": "Improvements from Baseline",
      "regressions_or_risks": "Risks and Trade-offs",
      "next_experiments": "Next Experiments",
  }
  for k, title in titles.items():
    v = result.get(k)
    if v is None:
      continue
    if isinstance(v, (list, tuple)):
      body = "\n".join(f"- {x}" for x in v)
    else:
      body = str(v)
    lines += [f"## {title}", "", body, ""]
  text = "\n".join(lines)
  with open(os.path.join(run_dir, "export", "analysis.md"), "w",
            encoding="utf-8") as f:
    f.write(text)
  logger.info("Analysis written to %s", os.path.join(run_dir, "export",
                                                     "analysis.md"))
  return result


def _respawn(classroom, students, cfgs, run_dir, instructions, cycle, logger):
  """Tops the classroom back up to its original size.

  New students start from the best solution found so far in this run (or
  the baseline), under a fresh name, so crossover partners never run out.
  """
  target = len(cfgs)
  if len(classroom) >= target:
    return classroom
  ranked = sorted(
      (st for st in students.values() if st.best_solution),
      key=lambda st: st.best_score, reverse=True)
  donor = ranked[0] if ranked else None
  added = 0
  while len(classroom) < target:
    base_cfg = cfgs[(len(students) + added) % len(cfgs)]
    name = f"{base_cfg.name}_r{cycle + 1}"
    cfg = dataclasses.replace(base_cfg, name=name)
    st = _LoggedStudentAgent(config=cfg, run_base_dir=run_dir)
    st.system_instruction = instructions
    if donor is not None:
      st.current_solution = donor.best_solution
      st.best_solution = donor.best_solution
      st.best_score = donor.best_score
      st.best_cycle = donor.best_cycle
      st.cycle_results = list(donor.cycle_results)
    else:
      st.current_solution = config.BASELINE_SOLUTION
    students[name] = st
    classroom.append(st)
    added += 1
  logger.info("Respawned %d students for cycle %d (donor: %s)", added,
              cycle + 1, donor.name if donor else "baseline")
  return classroom


# ============================================================================
# Watch mode: rerun when recordings change
# ============================================================================

STATE_FILE = os.path.join(TASK_DIR, ".last_fingerprint")


def _run_classroom(cycles: int, extra: list[str]) -> int:
  cmd = [
      sys.executable, "-m", "codegen_task.run_micro_gesture",
      f"--num_students={FLAGS.num_students}",
      f"--num_cycles={cycles}",
      f"--cv_folds={FLAGS.cv_folds}",
      f"--timeout_s={FLAGS.timeout_s}",
      f"--hil={FLAGS.hil}",
      f"--model_names={','.join(FLAGS.model_names)}",
      "--deploy",
      *extra,
  ]
  print(f"Auto-Watch triggering classroom run: {' '.join(cmd)}", flush=True)
  return subprocess.call(cmd)


def watch_main(argv_rest: list[str]) -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--interval", type=float, default=None)
  ap.add_argument("--quiet", type=float, default=None)
  ap.add_argument("--cycles", type=int, default=None)
  ap.add_argument("--once", action="store_true")
  ap.add_argument("--data_dir", default=None)
  args, extra = ap.parse_known_args(argv_rest)
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

  interval = args.interval if args.interval is not None else FLAGS.watch_interval
  quiet = args.quiet if args.quiet is not None else FLAGS.watch_quiet
  cycles = args.cycles if args.cycles is not None else FLAGS.num_cycles
  once = bool(args.once or FLAGS.watch_once)
  data_dir = args.data_dir or FLAGS.data_dir or data_loader.default_data_dir()

  last = ""
  if os.path.isfile(STATE_FILE):
    with open(STATE_FILE, encoding="utf-8") as f:
      last = f.read().strip()
  cur_fp = data_loader.dataset_fingerprint(data_dir)
  if not last:
    last = cur_fp
    with open(STATE_FILE, "w", encoding="utf-8") as f:
      f.write(last)
  print(
      f"Auto-Watch active on {data_dir} (baseline fingerprint={last}, "
      f"cycles={cycles}, quiet={quiet:.0f}s, poll={interval:.0f}s). "
      f"Record new gestures in Playground to trigger an automatic run.",
      flush=True,
  )
  pending_since: float | None = None
  pending_fp = ""
  while True:
    fp = data_loader.dataset_fingerprint(data_dir)
    if fp != last:
      if fp != pending_fp:
        pending_fp, pending_since = fp, time.time()
        print(
            f"Dataset changed ({last or 'none'} -> {fp}); waiting {quiet:.0f}s"
            " quiet period before launching classroom...",
            flush=True,
        )
      elif pending_since and time.time() - pending_since >= quiet:
        rc = _run_classroom(cycles, extra)
        if rc == 0:
          last = fp
          with open(STATE_FILE, "w", encoding="utf-8") as f:
            f.write(fp)
        pending_fp, pending_since = "", None
        if once:
          return
    elif once:
      print(f"No change ({fp})", flush=True)
      return
    time.sleep(interval)


def _extract_solution_code_by_id(sol_id: str) -> tuple[str, str]:
  """Returns (resolved_solution_id, python_code) from solutions/ or run_dir."""
  target_id = (sol_id or "").strip()
  lb_path = os.path.join(TASK_DIR, "leaderboard.tsv")
  df = pd.read_csv(lb_path, sep="\t") if os.path.isfile(lb_path) else pd.DataFrame()
  if (not target_id or target_id == "best") and not df.empty and "solution_id" in df:
    df_sorted = df.sort_values("test_overall", ascending=False)
    target_id = str(df_sorted.iloc[0]["solution_id"])
  md_path = os.path.join(SOLUTIONS_DIR, f"{target_id}.md")
  if os.path.isfile(md_path):
    with open(md_path, encoding="utf-8") as f:
      text = f.read()
    top = text.find("## Top Solution 1")
    start = text.find("```python", top if top >= 0 else 0)
    end = text.find("\n```", start + 9)
    if start >= 0 and end > start:
      code, _ = code_execution.extract_python_code(text[start:end + 4])
      return target_id, code
  if not df.empty and "solution_id" in df and "run_dir" in df:
    m = df[df["solution_id"].astype(str) == target_id]
    if not m.empty:
      rel = str(m.iloc[-1]["run_dir"])
      base = os.environ.get("CODEGEN_EXPERIMENTS_DIR",
                            os.path.join(REPO_DIR, "experiments"))
      for cand_dir in (rel, os.path.join(base, "micro_gesture", rel)):
        sol_py = os.path.join(cand_dir, "export", "solution.py")
        if os.path.isfile(sol_py):
          with open(sol_py, encoding="utf-8") as f:
            return target_id, f.read()
  raise FileNotFoundError(f"Solution code not found for '{target_id}'")


def retrain_solution_main(sol_id: str) -> None:
  """Re-fits a leaderboard solution on the current dataset and exports it."""
  splits = data_loader.load_splits(FLAGS.data_dir)
  deploy.configure(splits.class_names)
  resolved_id, code = _extract_solution_code_by_id(sol_id)
  print(
      f"Re-fitting leaderboard solution '{resolved_id}' on dataset"
      f" {splits.fingerprint} (train+val={len(splits.x_learning)},"
      f" test={len(splits.x_test)} recordings, {len(splits.class_names)}"
      f" classes: {splits.class_names})...",
      flush=True,
  )
  run_dir = os.path.join(_runs_base(), f"retrain_{resolved_id}")
  os.makedirs(run_dir, exist_ok=True)
  summary = _materialize_best(code, splits, run_dir, best_name=resolved_id)
  if summary is None:
    raise RuntimeError(f"Failed to materialize solution '{resolved_id}'")
  summary["solution_id"] = resolved_id
  art = summary["artifact"]
  tm = summary["test_metrics"]
  print(
      f"[retrain ok] {resolved_id}: test_overall={summary['test_overall_score']:.4f}"
      f" f1={tm.get('f1_macro', 0):.4f} wcr={tm.get('worst_class_recall', 0):.4f}"
      f" nfpr={tm.get('null_fpr', 0):.4f} kb={art['model_kb']:.2f}"
      f" mflops={art['mflops']:.3f} arena={art['arena_kb']:.2f} KB",
      flush=True,
  )
  os.makedirs(FIRMWARE_DIR, exist_ok=True)
  shutil.copy(
      os.path.join(run_dir, "export", "model_data.h"),
      os.path.join(FIRMWARE_DIR, "model_data.h"),
  )
  data_utils.save_dict_to_json(summary, DEPLOYED_JSON)
  exp_deploy_dir = os.path.join(REPO_DIR, "experiments", "deployed")
  os.makedirs(exp_deploy_dir, exist_ok=True)
  data_utils.save_dict_to_json(
      summary, os.path.join(exp_deploy_dir, "deployed_latest.json")
  )
  data_utils.save_dict_to_json(
      summary, os.path.join(exp_deploy_dir, "deployed.json")
  )
  print(
      f"Exported {resolved_id} to {os.path.join(FIRMWARE_DIR, 'model_data.h')}"
      f" and {DEPLOYED_JSON}",
      flush=True,
  )


# ============================================================================
# PROMPT.md rendering
# ============================================================================

ORIGIN_PROMPT = """\
## Origin: how this task was requested

Paraphrased from the conversation that produced `Python/codegen_task`, so
the task can be recreated from scratch with an AI agent or by hand.

> Clone https://github.com/google-research/codegen-classroom next to the
> on-device ML playground and review how it works: the student contract, the
> scoring loop, mutation strategies, and the `tasks/template/` layout.
>
> Build a classroom task called `micro_gesture` that evolves the playground's
> gesture classifier (XIAO nRF52840 Sense, LSM6DS3 at 100 Hz, recordings in
> `Python/data/<Label>-<n>.txt`) so it runs efficiently and accurately on the
> device under TensorFlow Lite Micro, and self-tunes.
>
> Constraints and preferences:
> - Minimal or zero changes to the codegen-classroom library; keep all task
>   code in the playground repo, reusing existing folders (`Python/`,
>   `Arduino/`) and keeping the original playground functional and its diff
>   small.
> - Students may use any library (TensorFlow, JAX, PyTorch), but the artifact
>   must be a TFLite model the firmware can run.
> - Custom scoring inside the task: accuracy plus on-device cost such as
>   latency, FLOPs, model size, arena, parameter count, int8 coverage, and
>   robustness metrics such as worst-class recall and Null false-trigger rate.
>   Include as many useful metrics and tips for the students as possible.
> - Include `gemini-4-argon` among the students.
> - Set it up to demonstrate hill climbing across runs and to rerun
>   automatically when new recordings arrive.
> - Prompt the human for more recordings when data is thin.
> - Keep a `PROMPT.md` summarizing the prompts so the task is reproducible.

Design decisions that followed from this request:

- Students attach the TFLite bytes to `y_pred_df.attrs["tflite_model"]`; a
  custom validator analyses the artifact in the parent thread and hands cost
  metrics to the scorer through a thread-local, so the library stays unmodified.
- The validator reruns the TFLite model on raw `X_val` and requires agreement
  with the returned predictions, which forces normalization into the graph.
- Splits are frozen per class by a hash of the file name, so the test set is
  stable as recordings accumulate; `solutions/best_solution.md` seeds the next
  run; `watch.py` reruns on dataset change; `--deploy` promotes to
  `Arduino/Codegen/model_data.h` only when the test score improves.
- A dedicated `Arduino/Codegen` sketch registers exactly the allowed ops,
  handles int8 or float I/O, and offers a `REPLAY` mode for on-device
  evaluation, leaving the upstream `Inference` sketch untouched.
"""

PROMPT_MD = os.path.join(TASK_DIR, "PROMPT.md")


def render_prompt() -> str:
  s = data_loader.load_splits()
  deploy.configure(s.class_names)
  instructions = config.build_student_instructions(
      class_names=s.class_names, train_rows=len(s.x_train),
      val_rows=len(s.x_val), final_eval_rows=len(s.x_learning),
      cv_folds=config.RUN_DEFAULTS["cv_folds"],
  )
  n_ref = len(config.STUDENT_REFINE_INSTRUCTION_OPTIONS)
  n_arch = len(config.ARCHETYPES)
  models = ", ".join(f"`{m}`" for m in config.DEFAULT_STUDENT_MODELS)
  rd = config.RUN_DEFAULTS
  return f"""# micro_gesture prompts

Generated by `python -m codegen_task.render_prompt` from
[`config.py`](config.py); regenerate after editing prompts or weights.
Dataset at render time: `{s.fingerprint}` (train {len(s.x_train)}, val
{len(s.x_val)}, test {len(s.x_test)}, classes {s.class_names}).

{ORIGIN_PROMPT}
## Prompt flow

| Step | Text sent | Source |
| :--- | :--- | :--- |
| Every call | System instruction below; unseeded students also get one of {n_arch} starting archetypes | `config.build_student_instructions`, `config.ARCHETYPES` |
| Cycle 0 | `STUDENT_INITIAL_PROMPT`, or the previous best solution (seeded half, after a budget check) | `config.py`, `solutions/best_solution.md` |
| Later cycles | Previous code, execution feedback, score breakdown, classroom summary plus run frontier and untried levers, one of {n_ref} refinement directions (plus analyzer-learned ones) | library, `knowledge.cycle_feedback`, `STUDENT_REFINE_INSTRUCTION_OPTIONS` |
| After `patience`={rd["patience"]} stale cycles | Library mutation prompt: self-best, best-peer crossover, or contrastive crossover | `agents.MutationStrategy` |
| Every response | Executed per fold in a sandbox; `deploy_validator` violations are returned verbatim as the retry traceback | `deploy.py` |

Students: {models}. Defaults: {rd["num_students"]} students,
{rd["num_cycles"]} cycles, {rd["cv_folds"]}-fold OOF scoring,
{rd["timeout_s"]} s per call, seed fraction {rd["seed_fraction"]}.

Initial prompt (cycle 0):

> {config.STUDENT_INITIAL_PROMPT}

Refinement directions ({n_ref}; see `config.STUDENT_REFINE_INSTRUCTION_OPTIONS`):

{chr(10).join(f"{i + 1}. {t.split(':')[0]}" for i, t in enumerate(config.STUDENT_REFINE_INSTRUCTION_OPTIONS))}

## System instruction (rendered)

{instructions.rstrip()}
"""


def render_prompt_main() -> None:
  text = render_prompt()
  with open(PROMPT_MD, "w", encoding="utf-8") as f:
    f.write(text)
  print(f"wrote {PROMPT_MD} ({len(text)} chars)")


class _SuppressExpandedCvSubsetFilter(logging.Filter):
  """Silences the expected 'CV folds predicted 467 of 600' message on expanded OOF."""

  def filter(self, record: logging.LogRecord) -> bool:
    return "CV folds predicted " not in record.getMessage()


def _archive_stale_leaderboard_rows(lb_path: str, cur_fp: str) -> None:
  """Moves leaderboard rows from older dataset fingerprints into experiments/archive_<fp>/."""
  if not os.path.isfile(lb_path):
    return
  try:
    df = pd.read_csv(lb_path, sep="\t")
  except Exception:  # pylint: disable=broad-exception-caught
    return
  if df.empty or "dataset" not in df:
    return
  stale_mask = df["dataset"].astype(str) != str(cur_fp)
  if not stale_mask.any():
    return
  stale_df = df[stale_mask]
  keep_df = df[~stale_mask]
  base = os.environ.get("CODEGEN_EXPERIMENTS_DIR",
                        os.path.join(REPO_DIR, "experiments"))
  for old_fp, grp in stale_df.groupby("dataset"):
    arch_dir = os.path.join(base, f"archive_{old_fp}")
    os.makedirs(arch_dir, exist_ok=True)
    arch_lb = os.path.join(arch_dir, "leaderboard.tsv")
    data_utils.submit_results_to_local_leaderboard(arch_lb, grp)
  keep_df.to_csv(lb_path, sep="\t", index=False)


def main(argv: list[str]) -> None:
  if FLAGS.prompt_only:
    render_prompt_main()
    return
  if FLAGS.watch:
    watch_main(argv[1:])
    return
  if FLAGS.retrain_solution:
    retrain_solution_main(FLAGS.retrain_solution)
    return
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "TF_NUM_INTRAOP_THREADS",
              "TF_NUM_INTEROP_THREADS"):
    os.environ.setdefault(var, "2")
  os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
  # Forked students inherit this; keeps re-fits reproducible on CPU.
  os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

  splits = data_loader.load_splits(FLAGS.data_dir)
  deploy.configure(splits.class_names)
  run_id = data_utils.generate_run_id()
  run_dir = os.path.join(_runs_base(), run_id)
  data_utils.create_dir(run_dir)
  if FLAGS.mock_students:  # report and knowledge skip marked runs
    open(os.path.join(run_dir, "MOCK"), "w", encoding="utf-8").close()
  log_level = logging.DEBUG if FLAGS.debug else logging.INFO
  logger = reporting.setup_logger(
      "codegen_classroom",
      level=log_level,
      log_file=os.path.join(run_dir, "run.log"),
  )
  logging.getLogger("codegen_classroom.student_learning").addFilter(
      _SuppressExpandedCvSubsetFilter()
  )
  if FLAGS.debug:
    absl_logging.set_verbosity(absl_logging.DEBUG)
    for noisy_logger in (
        "matplotlib", "matplotlib.font_manager", "matplotlib.pyplot",
        "PIL", "httpcore", "httpx", "urllib3", "google_genai",
    ):
      logging.getLogger(noisy_logger).setLevel(logging.WARNING)
  logger.info("run_dir %s", run_dir)
  logger.info("dataset %s: train=%d val=%d test=%d classes=%s",
              splits.fingerprint, len(splits.x_train), len(splits.x_val),
              len(splits.x_test), splits.class_names)

  metric_weights = dict(config.DEFAULT_METRIC_WEIGHTS)
  validator = functools.partial(
      deploy.deploy_validator, num_classes=len(splits.class_names)
  )
  prior = config.prompt_section(splits.fingerprint) if FLAGS.insights else ""
  instructions = config.build_student_instructions(
      class_names=splits.class_names,
      train_rows=len(splits.x_train),
      val_rows=len(splits.x_val),
      final_eval_rows=len(splits.x_learning),
      cv_folds=FLAGS.cv_folds,
      timeout_s=FLAGS.timeout_s,
      final_eval_timeout_s=FLAGS.final_eval_timeout_s,
      metric_weights=metric_weights,
      prior_insights=prior,
  )
  config.assert_no_recipes(instructions)
  with open(os.path.join(run_dir, "student_instructions.md"), "w",
            encoding="utf-8") as f:
    f.write(instructions)

  models = [FLAGS.model_names[i % len(FLAGS.model_names)]
            for i in range(FLAGS.num_students)]
  cfgs = student_learning.get_classroom_student_configs(
      system_instruction=instructions, model_names=models,
      max_output_tokens=FLAGS.max_output_tokens,
  )
  agent_cls = _MockStudent if FLAGS.mock_students else _LoggedStudentAgent
  students = {c.name: agent_cls(config=c, run_base_dir=run_dir) for c in cfgs}

  seed = _read_seed()
  if seed:
    seed = _validate_seed(seed, splits, logger) or config.BASELINE_SOLUTION
  n_seed = 0
  seeded_students: set[str] = set()
  if seed:
    n_seed = max(1, int(round(len(students) * FLAGS.seed_fraction)))
    for i, (st_name, st) in enumerate(students.items()):
      if i < n_seed:
        st.current_solution = seed
        seeded_students.add(st_name)
    logger.info("Seeded %d/%d students from the top leaderboard solution",
                n_seed, len(students))
  # Unseeded students start Cycle 0 from the Vanilla Original Playground
  # float32 MLP baseline and evolve along distinct directions from Cycle 1.
  for j, st in enumerate(list(students.values())[n_seed:]):
    st.current_solution = config.BASELINE_SOLUTION
    arch = config.ARCHETYPES[j % len(config.ARCHETYPES)]
    st.system_instruction = (
        instructions + "\n\n## Assigned starting direction\n\n"
        f"After evaluating the Cycle 0 Vanilla Original baseline, evolve"
        f" toward this direction: {arch}\n")
  if not seed:
    logger.info(
        "Initialized all %d students at Cycle 0 with Vanilla Original"
        " Playground float32 MLP baseline",
        len(students),
    )
  refine_options = list(config.STUDENT_REFINE_INSTRUCTION_OPTIONS)
  if FLAGS.insights:
    refine_options += config.learned_levers()

  cv_splits = None
  cyc_x, cyc_y, cyc_xv, cyc_yv = (
      splits.x_train, splits.y_train,
      splits.x_val_eval, splits.y_val_eval,
  )
  if FLAGS.cv_folds and FLAGS.cv_folds > 1:
    from sklearn import model_selection  # pylint: disable=import-outside-toplevel
    raw_x, raw_y = splits.x_learning, splits.y_learning
    labels = raw_y.iloc[:, 0].to_numpy()
    k = min(FLAGS.cv_folds, int(np.bincount(labels).min()))
    k = max(2, k)
    skf = model_selection.StratifiedKFold(n_splits=k, shuffle=True,
                                          random_state=0)
    null_idx = (
        splits.class_names.index("Null")
        if "Null" in splits.class_names else None
    )
    exp_x, exp_y = data_loader.expand_eval_frame(
        raw_x, raw_y, null_idx, seed=101
    )
    n_raw = len(raw_x)
    cyc_x = pd.concat([raw_x, exp_x], ignore_index=True)
    cyc_y = pd.concat([raw_y, exp_y], ignore_index=True)
    cyc_xv, cyc_yv = None, None
    exp_rec_ids = exp_x[data_loader.GROUP_COLUMN].to_numpy()
    cv_splits = []
    for tr_raw_idx, te_raw_idx in skf.split(raw_x, labels):
      te_recs = set(raw_x.iloc[te_raw_idx][data_loader.GROUP_COLUMN])
      te_exp_idx = (
          np.nonzero(np.isin(exp_rec_ids, list(te_recs)))[0] + n_raw
      )
      cv_splits.append((tr_raw_idx, te_exp_idx))
    logger.info(
        "Cycle scoring: %d-fold out-of-fold over %d recordings (%d expanded"
        " eval windows)",
        k, n_raw, len(exp_x),
    )

  classroom = list(students.values())
  history: MutableMapping[str, student_learning.ClassroomResults] = (
      collections.defaultdict(list)
  )
  active_metrics = ["overall_score"] + list(metric_weights)
  t_start = time.time()
  for cycle in range(FLAGS.num_cycles):
    if not classroom:
      break
    logger.info("=== Cycle %d (%d students) ===", cycle, len(classroom))
    feedback = scoring.get_classroom_feedback_string(
        cycle_num=cycle, current_classroom=classroom,
        classroom_results_history=history, metric_keys=active_metrics,
    )
    extra = config.cycle_feedback(students, refine_options)
    if extra:
      feedback = f"{feedback}\n\n{extra}"
    results = student_learning.run_parallel_classroom_learning_cycle(
        current_classroom=classroom,
        x_train=cyc_x, y_train=cyc_y, x_val=cyc_xv, y_val=cyc_yv,
        cv_splits=cv_splits, cv_sample_feedback_count=1,
        current_cycle=cycle,
        student_initial_prompt=config.STUDENT_INITIAL_PROMPT,
        student_function_name=config.STUDENT_FUNCTION_NAME,
        code_execution_max_retries=3 if cycle == 0 else 2,
        code_execution_timeout_seconds=FLAGS.timeout_s,
        metric_weights=metric_weights,
        early_stopping_patience=FLAGS.patience,
        student_refine_instruction_options=refine_options,
        max_mutation_count=FLAGS.max_mutations,
        mutation_strategy_options=[
            agents.MutationStrategy.SELF_BEST_CODE,
            agents.MutationStrategy.BEST_PEER_CROSSOVER,
            agents.MutationStrategy.CONTRASTIVE_PEER_CROSSOVER,
        ],
        validator=validator,
        classroom_feedback=feedback,
        num_workers=len(classroom),
        verbose_logging=FLAGS.debug,
    )
    classroom = student_learning.process_cycle_results(
        classroom, results, history, cycle
    )
    for st_name, st_obj in students.items():
      crs = getattr(st_obj, "cycle_results", [])
      for cr in crs:
        if getattr(cr, "cycle", None) == cycle:
          m = cr.metrics or {}
          sc = cr.overall_score
          if sc is not None:
            logger.info(
                "[cycle %d] %s: overall=%.4f f1=%.3f wcr=%.3f nfpr=%.3f"
                " kb=%.1f mflops=%.3f arena=%.1f",
                cycle, st_name, float(sc),
                float(m.get("f1_macro", float("nan"))),
                float(m.get("worst_class_recall", float("nan"))),
                float(m.get("null_fpr", float("nan"))),
                float(m.get("model_kb", float("nan"))),
                float(m.get("mflops", float("nan"))),
                float(m.get("arena_kb", float("nan"))),
            )
            plan_txt = ""
            raw_plan = str(getattr(cr, "plan", "") or "").strip()
            if raw_plan and raw_plan.lower() != "nan":
              plan_txt = raw_plan
            if not plan_txt:
              fb_str = str(getattr(cr, "execution_feedback", "") or "")
              for fb_ln in fb_str.splitlines():
                if fb_ln.strip().lower().startswith("hypothesis:"):
                  plan_txt = fb_ln.strip()
                  break
            if not plan_txt:
              code_str = str(getattr(cr, "code", "") or "")
              doc_m = re.search(r'"""(.*?)"""', code_str, re.DOTALL)
              if doc_m:
                plan_txt = doc_m.group(1).strip()
            mut_s = str(getattr(cr, "mutation_strategy", "") or "").strip()
            collab_s = str(getattr(cr, "collaborating_agent", "") or "").strip()
            prefix_parts = []
            if mut_s and mut_s.lower() != "nan":
              prefix_parts.append(mut_s)
            if collab_s and collab_s.lower() != "nan":
              prefix_parts.append(f"with {collab_s}")
            prefix = f"[{' '.join(prefix_parts)}] " if prefix_parts else ""
            if FLAGS.debug and plan_txt:
              logger.info("  plan: %s\n%s", prefix, plan_txt)
            elif plan_txt or prefix:
              logger.info(
                  "  plan: %s%s",
                  prefix,
                  plan_txt.replace("\n", " ")[:280],
              )
          else:
            err = getattr(cr, "error_message", "") or "failed"
            logger.info("[cycle %d] %s: FAILED (%s)",
                        cycle, st_name, str(err).splitlines()[-1][:200])
    if FLAGS.respawn and cycle + 1 < FLAGS.num_cycles:
      classroom = _respawn(classroom, students, cfgs, run_dir, instructions,
                           cycle, logger)
    for s in students.values():
      s.reset_history()
    if FLAGS.mock_students:
      break

  eval_path = os.path.join(run_dir, "final_eval_results.json")
  student_learning.run_parallel_classroom_final_evaluation(
      students=students,
      x_learning=splits.x_learning, y_learning=splits.y_learning,
      x_test_final=splits.x_test_eval, y_test_final=splits.y_test_eval,
      student_function_name=config.STUDENT_FUNCTION_NAME,
      metric_weights=metric_weights,
      code_execution_timeout_seconds=FLAGS.final_eval_timeout_s,
      enforce_timeout=True,
      strategies=["first", "best"],  # first vs best on test shows the climb
      validator=validator,
      num_workers=1,
      final_results_path=eval_path,
      predictions_dir=os.path.join(run_dir, "predictions"),
  )

  sections = reporting.build_run_report(
      run_dir, students=list(students.values()), metric_weights=metric_weights,
      top_n=2, worst_n=0,
  )
  reporting.write_report(run_dir, sections, precision=3)

  if not os.path.isfile(eval_path):
    logger.warning("No final evaluation results; nothing to export.")
    return
  with open(eval_path, encoding="utf-8") as f:
    eval_json = json.load(f)
  best_name, best_eval = student_learning.get_best_solution_from_eval_results(
      eval_json, metric_key="overall_score", solution_strategy="best"
  )
  if not best_name or best_name not in students:
    logger.warning("No valid final solution; nothing to export.")
    return
  best_code, _ = code_execution.extract_python_code(
      students[best_name].best_solution
  )
  logger.info("Best on test: %s overall=%.4f f1=%.3f", best_name,
              best_eval.get("overall_score", float("nan")),
              best_eval.get("metrics_f1_macro", float("nan")))

  summary = _materialize_best(best_code, splits, run_dir, best_name,
                              best_eval)
  if summary is None:
    logger.warning("Could not materialize the best artifact.")
    return
  lib_score = float(best_eval.get("overall_score", float("nan")))
  summary["final_eval_overall_score"] = lib_score
  summary["solution_id"] = _solution_id(len(cfgs), FLAGS.num_cycles)
  summary["retrain_delta"] = summary["test_overall_score"] - lib_score
  logger.info("Retrain delta (materialized - final eval): %+.4f",
              summary["retrain_delta"])

  os.makedirs(SOLUTIONS_DIR, exist_ok=True)
  analysis = None
  if FLAGS.analysis and not FLAGS.mock_students:
    try:
      analysis = _analyze_winner(students, best_name, best_code, best_eval,
                                 summary, instructions, run_dir, logger)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("Analysis step failed: %s", e, exc_info=True)
  if analysis:
    summary["analysis"] = analysis
    data_utils.save_dict_to_json(
        summary, os.path.join(run_dir, "export", "summary.json"))
    shutil.copy(os.path.join(run_dir, "export", "analysis.md"),
                os.path.join(SOLUTIONS_DIR, "best_analysis.md"))

  if FLAGS.insights and not FLAGS.mock_students:
    try:
      config.record_run(run_id, summary, analysis, students,
                           splits.fingerprint)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("Insight recording failed: %s", e)

  os.makedirs(SOLUTIONS_DIR, exist_ok=True)

  art = summary["artifact"]
  tm = summary["test_metrics"]
  sol_id = summary["solution_id"]
  raw_tags = (analysis or {}).get("model_tags", "")
  if isinstance(raw_tags, (list, tuple)):
    raw_tags = ", ".join(str(x) for x in raw_tags)
  tags_str = "mock" if FLAGS.mock_students else str(raw_tags or "").strip()
  row = {
      "timestamp": summary["timestamp"],
      "solution_id": sol_id,
      "dataset": splits.fingerprint,
      "n_train": len(splits.x_train), "n_val": len(splits.x_val),
      "n_test": len(splits.x_test),
      "test_overall": round(summary["test_overall_score"], 4),
      "test_f1_macro": round(tm.get("f1_macro", 0), 4),
      "test_wcr": round(tm.get("worst_class_recall", 0), 4),
      "test_bacc": round(tm.get("balanced_accuracy", 0), 4),
      "retrain_delta": round(summary["retrain_delta"], 4),
      "model_kb": round(art["model_kb"], 2),
      "mflops": round(art["mflops"], 4),
      "arena_kb": round(art["arena_kb"], 2),
      "host_ms": round(art["host_latency_ms"], 3),
      "int8": art["quantized"],
      "ops": "|".join(art["ops"]),
      "student": best_name,
      "cycles": FLAGS.num_cycles,
      "seeded": best_name in seeded_students,
      "tags": tags_str,
      "elapsed_s": round(time.time() - t_start, 1),
      "elo": 1500.0,
      "run_dir": run_dir,
  }
  if not FLAGS.mock_students:
    lb_path = os.path.join(TASK_DIR, "leaderboard.tsv")
    _archive_stale_leaderboard_rows(lb_path, splits.fingerprint)
    data_utils.submit_results_to_local_leaderboard(
        lb_path, pd.DataFrame([row])
    )
    try:
      sol_elos = report_lib.update_leaderboard_elo(lb_path)
      if sol_id in sol_elos:
        row["elo"] = sol_elos[sol_id]
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  promoted = _promote(summary, run_dir) if FLAGS.deploy else False
  analysis_md = ""
  if FLAGS.report:
    try:
      path = report_lib.build_report(
          run_dir, splits, hil=FLAGS.hil, promoted=promoted,
          figs_dir=os.path.join(SOLUTIONS_DIR, "plots"),
          fig_prefix=f"{sol_id}_", fig_rel="plots")
      analysis_md = report_lib.analysis_markdown()
      best_rep = os.path.join(SOLUTIONS_DIR, "best_report.md")
      if analysis_md and (promoted or not os.path.isfile(best_rep)):
        with open(best_rep, "w", encoding="utf-8") as f:
          f.write(analysis_md)
      logger.info("Report: %s", path)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("Report generation failed: %s", e, exc_info=True)
  if not FLAGS.mock_students:
    try:
      _write_solution(sol_id, run_dir, students, metric_weights, row,
                      analysis_md, logger)
      deployed = []
      if os.path.isfile(DEPLOYED_JSON):
        deployed = [str(data_utils.load_json_from_file(DEPLOYED_JSON)
                        .get("solution_id", ""))]
      _prune_solutions(keep_extra=[d for d in deployed if d])
    except Exception as e:  # pylint: disable=broad-exception-caught
      logger.warning("Solution export failed: %s", e, exc_info=True)
  logger.info("Export: %s", os.path.join(run_dir, "export"))


if __name__ == "__main__":
  app.run(main)
