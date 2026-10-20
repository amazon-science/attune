# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""workload.py step counts, the calibration file's measurement stamps and the verdict band.
Run: python3 -m pytest skill/scripts/tests"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "correctness"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibration  # noqa: E402
import fixtures  # noqa: E402
import parity_verdict  # noqa: E402
import workload  # noqa: E402


@pytest.fixture
def env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("AUTOTUNE_"):
            monkeypatch.delenv(k)
    return monkeypatch


def test_default_layout(env):
    wl = workload.resolve()
    assert (wl["num_steps"], wl["window_start"], wl["window_end"], wl["target_step"]) == (20, 5, 19, 19)
    assert wl["confirm_num_steps"] == 20 and workload.CHECK_STEPS == parity_verdict.STEPS
    pr = workload.profiled(wl)
    assert (pr["profile_setup_step"], pr["profile_warmup_step"], pr["profile_start"], pr["profile_end"]) == (20, 21, 22, 23)
    assert (pr["num_steps_profiled"], pr["target_step_profiled"]) == (23, 22)
    assert [workload.profiled(wl, k)["num_steps_profiled"] for k in (1, 2, 3)] == [23, 24, 25]


def test_capture_bounds_and_minimum_steps(env, capsys):
    env.setenv("AUTOTUNE_PROFILE_CAPTURE", "3")
    wl = workload.resolve()
    assert workload.profiled(wl)["profile_end"] == 25 and workload.profiled(wl, 1)["profile_end"] == 23
    for bad in ("0", "4"):
        env.setenv("AUTOTUNE_PROFILE_CAPTURE", bad)
        assert workload.resolve()["target_step"] == 19          # the unprofiled layout ignores the capture setting
        with pytest.raises(ValueError):
            workload.profiled(workload.resolve())
    env.delenv("AUTOTUNE_PROFILE_CAPTURE")
    with pytest.raises(ValueError, match="outside 1..3"):
        workload.profiled(wl, 4)
    env.setenv("AUTOTUNE_NUM_STEPS", "7")       # window 5-6: fewer than 3 measured steps
    with pytest.raises(ValueError, match="AUTOTUNE_NUM_STEPS=7 is below the minimum 8"):
        workload.resolve()
    env.delenv("AUTOTUNE_NUM_STEPS")
    env.setenv("AUTOTUNE_WINDOW_START", "25")   # window start past N-1 = 19 with N unset: the default is checked too
    with pytest.raises(ValueError, match="unset and its default 20 is below the minimum 28"):
        workload.resolve()
    env.setenv("AUTOTUNE_NUM_STEPS", "26")      # window 25-25: fewer than 3 measured steps
    with pytest.raises(ValueError, match="AUTOTUNE_NUM_STEPS=26 is below the minimum 28"):
        workload.resolve()
    env.setenv("AUTOTUNE_WINDOW_START", "17")   # 17-19 is the shortest window at N = 20
    env.delenv("AUTOTUNE_NUM_STEPS")
    assert (workload.resolve()["window_start"], workload.resolve()["window_end"]) == (17, 19)
    env.delenv("AUTOTUNE_WINDOW_START")
    env.setenv("AUTOTUNE_NUM_STEPS", "30")
    env.setenv("AUTOTUNE_NUM_STEPS_PROFILED", "32")
    wl = workload.resolve()
    assert wl["window_end"] == 29 and workload.profiled(wl)["profile_start"] == 32
    assert "AUTOTUNE_NUM_STEPS_PROFILED is no longer read" in capsys.readouterr().err


def test_run_length_is_the_operators(env):
    """Shorter and longer ranked runs keep the window START..N-1; a confirm run always holds the check's 20 steps."""
    env.setenv("AUTOTUNE_NUM_STEPS", "8")       # the shortest run at the default window start: steps 5-7
    wl = workload.resolve()
    assert (wl["window_start"], wl["window_end"], wl["target_step"], wl["confirm_num_steps"]) == (5, 7, 7, 20)
    env.setenv("AUTOTUNE_NUM_STEPS", "12")
    env.setenv("AUTOTUNE_WINDOW_START", "2")
    wl = workload.resolve()
    assert (wl["num_steps"], wl["window_start"], wl["window_end"], wl["confirm_num_steps"]) == (12, 2, 11, 20)
    pr = workload.profiled(wl)                  # the profiler still runs after the measured steps
    assert (pr["profile_setup_step"], pr["profile_start"], pr["num_steps_profiled"]) == (12, 14, 15)
    env.delenv("AUTOTUNE_WINDOW_START")
    env.setenv("AUTOTUNE_NUM_STEPS", "30")      # longer runs are their own confirm length
    wl = workload.resolve()
    assert (wl["window_end"], wl["confirm_num_steps"]) == (29, 30)


def test_cli_reports_a_bad_capture_without_failing(env):
    e = dict(os.environ, AUTOTUNE_PROFILE_CAPTURE="5")
    p = subprocess.run([sys.executable, str(SCRIPTS / "workload.py")], capture_output=True, text=True, env=e)
    assert p.returncode == 0 and json.loads(p.stdout)["profile_error"].startswith("profile capture 5")


def _calibration(**kw):
    d = {"runs": [{"nodes": ["n%d" % i]} for i in range(5)], "denominator": 100.0, "band": 0.03,
         "num_steps": 20, "window": [5, 19]}
    d.update(kw)
    return d


def test_calibration_must_match_the_measurement(env):
    wl = workload.resolve()
    assert calibration.problems(_calibration(), wl) == []
    stale = calibration.problems(_calibration(num_steps=14, window=[4, 13]), wl)
    assert any("the campaign measures num_steps 20, window [5, 19]" in x and "calibration.py build" in x for x in stale)
    unstamped = calibration.problems({k: v for k, v in _calibration().items() if k != "num_steps"}, wl)
    assert any("no num_steps/window stamps" in x and "calibration.py stamp" in x for x in unstamped)


def test_calibration_cli(env, tmp_path):
    f = tmp_path / "cal.json"
    f.write_text(json.dumps(_calibration()))
    e = dict(os.environ, AUTOTUNE_CALIBRATION=str(f))
    p = subprocess.run([sys.executable, str(SCRIPTS / "calibration.py")], capture_output=True, text=True, env=e)
    assert p.returncode == 0 and json.loads(p.stdout)["window"] == [5, 19], p.stderr
    f.write_text(json.dumps(_calibration(num_steps=14, window=[4, 13])))
    p = subprocess.run([sys.executable, str(SCRIPTS / "calibration.py")], capture_output=True, text=True, env=e)
    assert p.returncode == 1 and "measured with num_steps 14" in p.stderr


def _cal(*args, **extra):
    e = dict(os.environ, **extra)
    return subprocess.run([sys.executable, str(SCRIPTS / "calibration.py"), *args], capture_output=True, text=True, env=e)


def test_calibration_build_scores_and_stamps(env, tmp_path):
    reps = []
    for i, tok in enumerate((1000.0, 1010.0, 990.0, 1000.0, 1005.0)):
        log = tmp_path / ("cal%d.log" % i)
        log.write_text(fixtures.train_log(steps=20).replace("'perf/actor_train_tok_per_s': 1000.0",
                                                            "'perf/actor_train_tok_per_s': %r" % tok))
        reps += ["--replicate", "alice-cal%d" % i, str(log), "node-%da,node-%db" % (i, i)]
    out = tmp_path / "cal.json"
    p = _cal("build", "--out", str(out), "--num-gpus", "16", *reps)
    assert p.returncode == 0, p.stderr
    d = json.loads(out.read_text())
    toks = [round(tok * 9.0 / 10.0 / 16, 2) for tok in (1000.0, 1010.0, 990.0, 1000.0, 1005.0)]   # read_metrics rounds
    assert (d["num_steps"], d["window"], len(d["runs"])) == (20, [5, 19], 5)
    assert d["denominator"] == pytest.approx(sum(toks) / 5) and d["band"] == 0.03   # 3 sigma / mean < 3%
    assert _cal(AUTOTUNE_CALIBRATION=str(out)).returncode == 0
    p = _cal("build", "--out", str(tmp_path / "short.json"), "--num-gpus", "16", *reps[:8])   # two replicates
    assert p.returncode == 1 and "need >= 5" in p.stderr and not (tmp_path / "short.json").exists()
    (tmp_path / "trunc.log").write_text(fixtures.train_log(steps=14))
    p = _cal("build", "--out", str(tmp_path / "bad.json"), "--num-gpus", "16", *reps[:-4],
             "--replicate", "alice-trunc", str(tmp_path / "trunc.log"), "node-9")
    assert p.returncode == 1 and "alice-trunc: no valid window 5-19" in p.stderr and not (tmp_path / "bad.json").exists()


def test_calibration_stamp_migrates_without_rewriting_history(env, tmp_path):
    f = tmp_path / "old.json"
    f.write_text(json.dumps({k: v for k, v in _calibration(window=[4, 13]).items() if k != "num_steps"}))
    p = _cal(AUTOTUNE_CALIBRATION=str(f))
    assert p.returncode == 1 and "no num_steps/window stamps" in p.stderr
    p = _cal("stamp", str(f), "--num-steps", "20", "--window", "5", "19")
    assert p.returncode == 1 and "already records window [4, 13]" in p.stderr
    p = _cal("stamp", str(f), "--num-steps", "14", "--window", "4", "13")
    assert p.returncode == 0 and "still unusable" in p.stdout, p.stderr
    assert json.loads(f.read_text())["num_steps"] == 14
    p = _cal(AUTOTUNE_CALIBRATION=str(f))
    assert p.returncode == 1 and "measured with num_steps 14, window [4, 13]" in p.stderr
    g = tmp_path / "unstamped.json"
    g.write_text(json.dumps({k: v for k, v in _calibration().items() if k not in ("num_steps", "window")}))
    assert _cal("stamp", str(g), "--num-steps", "20", "--window", "5", "19").returncode == 0
    assert _cal(AUTOTUNE_CALIBRATION=str(g)).returncode == 0


def test_the_band_is_the_operators(env, tmp_path):
    p = _cal()
    assert p.returncode == 0 and json.loads(p.stdout) == {"band": 0.03, "denominator": None,
                                                          "source": "default (no calibration file)"}, p.stderr
    p = _cal(AUTOTUNE_BAND="0.05")
    assert p.returncode == 0 and json.loads(p.stdout)["band"] == 0.05, p.stderr
    assert json.loads(p.stdout)["source"] == "AUTOTUNE_BAND (no calibration file)"
    p = _cal(AUTOTUNE_BAND="0.01")              # used as given: only a calibration file's band is floored at 3%
    assert p.returncode == 0 and json.loads(p.stdout)["band"] == 0.01, p.stderr
    for bad in ("5", "0", "-0.01", "1", "nan", "five"):
        p = _cal(AUTOTUNE_BAND=bad)
        assert p.returncode == 1 and "AUTOTUNE_BAND=%r is not a fraction" % bad in p.stderr, (bad, p.stderr)
    f = tmp_path / "cal.json"                   # a calibration file's band replaces it
    f.write_text(json.dumps(_calibration(band=0.045)))
    p = _cal(AUTOTUNE_CALIBRATION=str(f), AUTOTUNE_BAND="0.02")
    assert p.returncode == 0 and json.loads(p.stdout)["band"] == 0.045, p.stderr
    assert "the calibration file's band applies; AUTOTUNE_BAND=0.02 is not used" in p.stderr
