"""Calibration POMDP module for Sphero tracker.

This defines a tiny active inference agent over calibration quality.
Place this file next to `sphero_track_test_v2.py` (e.g. in the `scripts/` folder).
"""

from enum import IntEnum
import math
import numpy as np

from pymdp import utils
from pymdp.agent import Agent


class CalibState(IntEnum):
    """Hidden calibration states."""
    OK = 0       # calibration good
    DRIFT = 1    # mild drift
    BAD = 2      # badly off


class CalibObs(IntEnum):
    """Discrete observation categories for calibration."""
    ERR_LOW = 0
    ERR_MED = 1
    ERR_HIGH = 2


class CalibAction(IntEnum):
    """Meta-actions that affect how aggressively we move."""
    NORMAL = 0    # full speed, standard margins
    CAUTION = 1   # reduced speed
    # RECAL = 2   # we can add a recalibration action later


def discretize_calib_obs(prediction_error_px: float,
                         stuck_frames: int,
                         safety_recent: bool) -> int:
    """Map continuous measures into a discrete observation index.

    Args
    ----
    prediction_error_px : float
        Current prediction/smoothing error in pixels.
    stuck_frames : int
        How many consecutive frames we've appeared stuck.
    safety_recent : bool
        Whether we were recently in a safety / boundary-avoidance state.

    Returns
    -------
    int
        Index into CalibObs (ERR_LOW, ERR_MED, ERR_HIGH).
    """
    # Treat non-finite prediction errors as medium
    if prediction_error_px is None or not math.isfinite(prediction_error_px):
        prediction_error_px = 20.0

    # If safety has been active recently, or we're stuck, treat as high error
    if safety_recent or stuck_frames >= 60 or prediction_error_px >= 40.0:
        return int(CalibObs.ERR_HIGH)

    if prediction_error_px >= 10.0:
        return int(CalibObs.ERR_MED)

    return int(CalibObs.ERR_LOW)


def build_calibration_agent() -> Agent:
    """Build a 1-factor POMDP agent over calibration quality.

    States:    OK, DRIFT, BAD
    Obs:       LOW, MED, HIGH error
    Actions:   NORMAL, CAUTION

    The agent uses expected free energy internally to score policies.
    """
    n_states = [len(CalibState)]     # 3
    n_obs = [len(CalibObs)]          # 3
    n_controls = [len(CalibAction)]  # 2

    # A: observation model p(o | s)
    A = utils.obj_array(1)
    A[0] = np.zeros((n_obs[0], n_states[0]))

    # Columns = states, rows = observations
    # OK: mostly low error
    A[0][CalibObs.ERR_LOW,  CalibState.OK]   = 0.75
    A[0][CalibObs.ERR_MED,  CalibState.OK]   = 0.20
    A[0][CalibObs.ERR_HIGH, CalibState.OK]   = 0.05

    # DRIFT: medium and some high
    A[0][CalibObs.ERR_LOW,  CalibState.DRIFT] = 0.20
    A[0][CalibObs.ERR_MED,  CalibState.DRIFT] = 0.55
    A[0][CalibObs.ERR_HIGH, CalibState.DRIFT] = 0.25

    # BAD: mostly high
    A[0][CalibObs.ERR_LOW,  CalibState.BAD]   = 0.05
    A[0][CalibObs.ERR_MED,  CalibState.BAD]   = 0.25
    A[0][CalibObs.ERR_HIGH, CalibState.BAD]   = 0.70

    # Normalise columns
    A[0] = A[0] / A[0].sum(axis=0, keepdims=True)

    # B: transition model p(s' | s, a), shape (n_states, n_states, n_controls)
    B = utils.obj_array(1)
    B[0] = np.zeros((n_states[0], n_states[0], n_controls[0]))

    # Under NORMAL: drift happens faster
    # From OK
    B[0][CalibState.OK,    CalibState.OK,    CalibAction.NORMAL] = 0.96
    B[0][CalibState.DRIFT, CalibState.OK,    CalibAction.NORMAL] = 0.03
    B[0][CalibState.BAD,   CalibState.OK,    CalibAction.NORMAL] = 0.01
    # From DRIFT
    B[0][CalibState.OK,    CalibState.DRIFT, CalibAction.NORMAL] = 0.03
    B[0][CalibState.DRIFT, CalibState.DRIFT, CalibAction.NORMAL] = 0.87
    B[0][CalibState.BAD,   CalibState.DRIFT, CalibAction.NORMAL] = 0.10
    # From BAD
    B[0][CalibState.OK,    CalibState.BAD,   CalibAction.NORMAL] = 0.00
    B[0][CalibState.DRIFT, CalibState.BAD,   CalibAction.NORMAL] = 0.10
    B[0][CalibState.BAD,   CalibState.BAD,   CalibAction.NORMAL] = 0.90

    # Under CAUTION: drift is slower
    # From OK
    B[0][CalibState.OK,    CalibState.OK,    CalibAction.CAUTION] = 0.98
    B[0][CalibState.DRIFT, CalibState.OK,    CalibAction.CAUTION] = 0.02
    B[0][CalibState.BAD,   CalibState.OK,    CalibAction.CAUTION] = 0.00
    # From DRIFT
    B[0][CalibState.OK,    CalibState.DRIFT, CalibAction.CAUTION] = 0.10
    B[0][CalibState.DRIFT, CalibState.DRIFT, CalibAction.CAUTION] = 0.85
    B[0][CalibState.BAD,   CalibState.DRIFT, CalibAction.CAUTION] = 0.05
    # From BAD
    B[0][CalibState.OK,    CalibState.BAD,   CalibAction.CAUTION] = 0.02
    B[0][CalibState.DRIFT, CalibState.BAD,   CalibAction.CAUTION] = 0.08
    B[0][CalibState.BAD,   CalibState.BAD,   CalibAction.CAUTION] = 0.90

    # Normalise over next-state rows for each (s, a)
    for a in range(n_controls[0]):
        B[0][:, :, a] = B[0][:, :, a] / B[0][:, :, a].sum(axis=0, keepdims=True)

    # C: preferences over observations (we like low error)
    C = utils.obj_array(1)
    c_vec = np.array([0.6, 0.3, 0.1])  # LOW, MED, HIGH
    C[0] = np.log(c_vec / c_vec.sum())

    # D: prior over states – optimistic
    D = utils.obj_array(1)
    D[0] = np.array([0.8, 0.15, 0.05])  # OK, DRIFT, BAD

    agent = Agent(A=A, B=B, C=C, D=D)
    return agent
