import ast
import math
import os
from pathlib import Path

import pytest

from conftest import load

cd = load("schedules.crossdistill")

TOL = 1e-8
SIGMA_MAX = 1600.0
START = SIGMA_MAX / (SIGMA_MAX + 1.0)

# Reference values from upstream SparkDiffusion @ ad6b65b
# (sparkdiffusion/inference/wan2pt1_t2v_distilled_infer.py, MID_T_BY_NSTEPS).
REFERENCE = {
    3: [START, 0.933781, 0.852895, 0.0],
    4: [START, 0.933781, 0.852895, 0.608979, 0.0],
}


def assert_close_seq(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert abs(x - y) <= TOL, (a, b)


def test_rf_start_matches_sigma_max_formula():
    assert abs(cd.rf_start(1600) - 0.999375390381) < 1e-12
    assert abs(cd.rf_start() - START) <= TOL


@pytest.mark.parametrize("steps", [3, 4])
def test_exact_reference_schedules(steps):
    assert_close_seq(cd.crossdistill_timesteps(steps), REFERENCE[steps])


def test_three_step_is_prefix_of_four_step():
    assert cd.crossdistill_timesteps(3)[:3] == cd.crossdistill_timesteps(4)[:3]


@pytest.mark.parametrize("steps", cd.SUPPORTED_STEPS)
def test_schedule_shape_and_monotonic(steps):
    ts = cd.crossdistill_timesteps(steps)
    assert len(ts) == steps + 1
    assert ts[-1] == 0.0
    assert all(a > b for a, b in zip(ts, ts[1:]))
    assert 0.0 < ts[0] < 1.0


@pytest.mark.parametrize("steps", [0, 5, 6, 7, 9, -1])
def test_unsupported_steps_raise(steps):
    with pytest.raises(ValueError):
        cd.crossdistill_timesteps(steps)


def test_invalid_sigma_max():
    with pytest.raises(ValueError):
        cd.rf_start(0)


def test_trigflow_origin_of_knots():
    """Upstream knots are TrigFlow times mapped by rf = sin t / (cos t + sin t)."""
    for knot in cd.MID_T_BY_NSTEPS[8]:
        t = math.atan2(knot, 1.0 - knot)  # inverse mapping
        assert abs(math.sin(t) / (math.cos(t) + math.sin(t)) - knot) < 1e-12


def test_network_timesteps_drop_terminal_zero():
    assert cd.network_timesteps(cd.crossdistill_timesteps(3)) == pytest.approx(
        (START * 1000, 933.781, 852.895), abs=1e-6
    )


def test_wan22_split_and_schedule():
    assert cd.wan22_split_steps(4) == (2, 2)
    assert cd.wan22_split_steps(3) == (1, 2)
    high, low = cd.wan22_timesteps(2, 2)
    assert_close_seq(high, [START, 0.933781, 0.875])
    assert_close_seq(low, [0.875, 0.608979, 0.0])
    with pytest.raises(ValueError):
        cd.wan22_timesteps(3, 2)
    with pytest.raises(ValueError):
        cd.wan22_split_steps(1)


def test_describe_mentions_values():
    assert "0.608979" in cd.describe(4)


def _upstream_repo():
    path = os.environ.get("SPARKDIFFUSION_REPO")
    return Path(path) if path and Path(path).is_dir() else None


def _literal_dicts(source: str, name: str):
    """Yield literal values assigned to ``name`` anywhere in a module (incl. ``if __name__`` blocks)."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            yield ast.literal_eval(node.value)


@pytest.mark.skipif(_upstream_repo() is None, reason="set SPARKDIFFUSION_REPO to compare with upstream source")
@pytest.mark.parametrize("entry", ["wan2pt1_t2v_distilled_infer.py", "wan2pt1_i2v_distilled_infer.py"])
def test_matches_upstream_source(entry):
    src = (_upstream_repo() / "sparkdiffusion" / "inference" / entry).read_text(encoding="utf-8")
    found = list(_literal_dicts(src, "MID_T_BY_NSTEPS"))
    assert found, "MID_T_BY_NSTEPS not found upstream"
    for upstream in found:
        assert set(upstream) == set(cd.MID_T_BY_NSTEPS)
        for k, v in upstream.items():
            assert_close_seq(list(v), list(cd.MID_T_BY_NSTEPS[k]))


@pytest.mark.skipif(_upstream_repo() is None, reason="set SPARKDIFFUSION_REPO to compare with upstream source")
def test_wan22_matches_upstream_source():
    src = (_upstream_repo() / "sparkdiffusion" / "inference" / "wan2pt2_t2v_distilled_infer.py").read_text(
        encoding="utf-8"
    )
    high = next(_literal_dicts(src, "MID_T_HIGH"))
    low = next(_literal_dicts(src, "MID_T_LOW"))
    for k, v in high.items():
        assert_close_seq(list(v), list(cd.WAN22_MID_T_HIGH[k]))
    for k, v in low.items():
        assert_close_seq(list(v), list(cd.WAN22_MID_T_LOW[k]))
    assert "default=0.875" in src.replace(" ", "")
