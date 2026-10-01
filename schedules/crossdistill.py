"""CrossDistill few-step timestep schedules used by SparkDiffusion.

The values below are transcribed from the upstream SparkDiffusion inference
entry points (``sparkdiffusion/inference/wan2pt1_t2v_distilled_infer.py``,
``wan2pt1_i2v_distilled_infer.py`` and ``wan2pt2_t2v_distilled_infer.py``).
They live in the rectified-flow (RF) domain, where ``t = 1`` is pure noise and
``t = 0`` is the clean sample. Upstream derives them from legacy TrigFlow
knots via ``rf = sin(t) / (cos(t) + sin(t))``.

Sampling procedure reproduced by upstream (and by ComfyUI's Euler sampler on a
flow model with ``shift = 1.0`` and ``multiplier = 1000``)::

    x = noise * t[0]
    for t_cur, t_next in zip(t[:-1], t[1:]):
        v = net(x, timestep=t_cur * 1000)
        x = x + (t_next - t_cur) * v

This module is pure Python (no torch import) so it can be imported during
ComfyUI start-up and in model-free tests.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

#: Upstream default ``--sigma_max`` for every distilled entry point.
SIGMA_MAX_DEFAULT: float = 1600.0

#: Upstream ``--t_scaling_factor`` (RF time -> network timestep).
T_SCALING_FACTOR: float = 1000.0

#: Intermediate RF knots per step count (Wan 2.1 T2V and I2V).
#: Source: ``MID_T_BY_NSTEPS`` in the upstream Wan 2.1 distilled entry points.
MID_T_BY_NSTEPS: Dict[int, Tuple[float, ...]] = {
    1: (),
    2: (0.933781,),
    3: (0.933781, 0.852895),
    4: (0.933781, 0.852895, 0.608979),
    8: (0.933781, 0.852895, 0.782708, 0.690833, 0.583052, 0.457197, 0.297157),
}

#: Wan 2.2 A14B MoE split: RF boundary between the high- and low-noise experts.
WAN22_BOUNDARY: float = 0.875

#: Wan 2.2 high-noise expert knots (covers ``[rf_start, boundary]``).
WAN22_MID_T_HIGH: Dict[int, Tuple[float, ...]] = {
    1: (),
    2: (0.933781,),
}

#: Wan 2.2 low-noise expert knots (covers ``[boundary, 0]``).
WAN22_MID_T_LOW: Dict[int, Tuple[float, ...]] = {
    1: (),
    2: (0.608979,),
    3: (0.720057, 0.507301),
    4: (0.720057, 0.507301, 0.297157),
}

SUPPORTED_STEPS: Tuple[int, ...] = tuple(sorted(MID_T_BY_NSTEPS))


def rf_start(sigma_max: float = SIGMA_MAX_DEFAULT) -> float:
    """Return the first RF time, ``sigma_max / (sigma_max + 1)``."""
    if not sigma_max > 0:
        raise ValueError(f"sigma_max must be positive, got {sigma_max!r}")
    return sigma_max / (sigma_max + 1.0)


def crossdistill_timesteps(num_steps: int, sigma_max: float = SIGMA_MAX_DEFAULT) -> Tuple[float, ...]:
    """Return the full RF schedule ``[rf_start, *mid_t, 0.0]`` for Wan 2.1.

    The returned tuple has ``num_steps + 1`` entries; consecutive pairs are the
    ``(t_cur, t_next)`` Euler steps.
    """
    if num_steps not in MID_T_BY_NSTEPS:
        raise ValueError(
            f"Unsupported CrossDistill step count {num_steps!r}; upstream supports {list(SUPPORTED_STEPS)}"
        )
    return (rf_start(sigma_max), *MID_T_BY_NSTEPS[num_steps], 0.0)


def wan22_split_steps(num_steps: int) -> Tuple[int, int]:
    """Split a total step count across the Wan 2.2 experts like upstream does.

    Mirrors ``STEPS_HIGH=$((NUM_STEPS / 2)); STEPS_LOW=$((NUM_STEPS - STEPS_HIGH))``
    in ``scripts/inference/eval_student_2pt2_distilled.sh``.
    """
    if num_steps < 2:
        raise ValueError("Wan 2.2 distilled inference needs at least 2 steps (one per expert)")
    high = num_steps // 2
    return high, num_steps - high


def wan22_timesteps(
    num_steps_high: int,
    num_steps_low: int,
    sigma_max: float = SIGMA_MAX_DEFAULT,
    boundary: float = WAN22_BOUNDARY,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Return ``(high, low)`` RF schedules for the Wan 2.2 A14B experts.

    Upstream falls back to a default knot list for unknown step counts; this
    function is stricter and raises instead, so a typo cannot silently change
    the schedule.
    """
    if num_steps_high not in WAN22_MID_T_HIGH:
        raise ValueError(f"Unsupported Wan 2.2 high-noise step count {num_steps_high!r}")
    if num_steps_low not in WAN22_MID_T_LOW:
        raise ValueError(f"Unsupported Wan 2.2 low-noise step count {num_steps_low!r}")
    high = (rf_start(sigma_max), *WAN22_MID_T_HIGH[num_steps_high], boundary)
    low = (boundary, *WAN22_MID_T_LOW[num_steps_low], 0.0)
    return high, low


def network_timesteps(rf_times: Sequence[float], scale: float = T_SCALING_FACTOR) -> Tuple[float, ...]:
    """Map RF times of the *evaluated* steps to the timestep values fed to the DiT.

    The terminal ``0.0`` is never evaluated, so it is dropped.
    """
    return tuple(float(t) * scale for t in rf_times[:-1])


def describe(num_steps: int, sigma_max: float = SIGMA_MAX_DEFAULT) -> str:
    """Human-readable one-line schedule description for logs and metadata."""
    ts = crossdistill_timesteps(num_steps, sigma_max)
    return f"crossdistill-{num_steps}step rf=[" + ", ".join(f"{t:.6f}" for t in ts) + "]"
