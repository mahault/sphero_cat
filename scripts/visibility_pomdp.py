"""
visibility_pomdp.py

Small POMDP over target visibility / occlusion for the Sphero tracker,
implemented with infer-actively pymdp.

Hidden states:
    VISIBLE   - target clearly visible
    OCCLUDED  - likely nearby but temporarily missing
    GONE      - likely left the scene

Observations:
    SEE_TARGET     - YOLO segmentation sees a target
    RECENTLY_LOST  - target not seen for a short time
    LONG_LOST      - target not seen for a longer time

Actions:
    TRACK  - behave as usual (move towards center if no target)
    WAIT   - stay in place while waiting for reappearance

This module is intentionally small and interpretable. It is used in
sphero_track_test_v2.py to decide whether to WAIT vs TRACK when the
target is currently not visible.
"""

from enum import IntEnum
import numpy as np

from pymdp import utils
from pymdp.agent import Agent


class VisState(IntEnum):
    """Hidden visibility states."""
    VISIBLE = 0
    OCCLUDED = 1
    GONE = 2


class VisObs(IntEnum):
    """Discrete observation categories."""
    SEE_TARGET = 0
    RECENTLY_LOST = 1
    LONG_LOST = 2


class VisAction(IntEnum):
    """Meta-actions when target is not currently visible."""
    TRACK = 0   # behave as usual: move to center when no target
    WAIT = 1    # stay in place and wait (if occluded)


def discretize_visibility_obs(target_found: bool,
                              target_lost_frames: int,
                              recent_thresh: int = 40) -> int:
    """
    Map 'target_found' + how long it's been missing into an observation index.

    Parameters
    ----------
    target_found : bool
        Whether detect_target() found a target this frame.
    target_lost_frames : int
        Number of consecutive frames with target_found == False.
    recent_thresh : int
        Number of frames considered 'recently lost'. At ~20 Hz,
        40 frames ~ 2 seconds.

    Returns
    -------
    int
        Index into VisObs (SEE_TARGET, RECENTLY_LOST, LONG_LOST).
    """
    if target_found:
        return int(VisObs.SEE_TARGET)
    if target_lost_frames < recent_thresh:
        return int(VisObs.RECENTLY_LOST)
    return int(VisObs.LONG_LOST)


def build_visibility_agent() -> Agent:
    """
    Build a 1-factor POMDP agent over target visibility.

    States:    VISIBLE, OCCLUDED, GONE
    Obs:       SEE_TARGET, RECENTLY_LOST, LONG_LOST
    Actions:   TRACK, WAIT
    """
    n_states = [len(VisState)]     # 3
    n_obs = [len(VisObs)]          # 3
    n_controls = [len(VisAction)]  # 2

    # A: observation model p(o | s)
    A = utils.obj_array(1)
    A[0] = np.zeros((n_obs[0], n_states[0]))

    # Columns = states, rows = observations
    #
    # If state = VISIBLE: usually see target
    A[0][VisObs.SEE_TARGET,     VisState.VISIBLE]   = 0.92
    A[0][VisObs.RECENTLY_LOST,  VisState.VISIBLE]   = 0.07
    A[0][VisObs.LONG_LOST,      VisState.VISIBLE]   = 0.01

    # If state = OCCLUDED:
    #   We often get RECENTLY_LOST, but not as aggressively as before,
    #   and LONG_LOST is a bit more common to avoid "sticky occlusion".
    A[0][VisObs.SEE_TARGET,     VisState.OCCLUDED]  = 0.30
    A[0][VisObs.RECENTLY_LOST,  VisState.OCCLUDED]  = 0.50
    A[0][VisObs.LONG_LOST,      VisState.OCCLUDED]  = 0.20

    # If state = GONE: mostly long-lost
    A[0][VisObs.SEE_TARGET,     VisState.GONE]      = 0.05
    A[0][VisObs.RECENTLY_LOST,  VisState.GONE]      = 0.15
    A[0][VisObs.LONG_LOST,      VisState.GONE]      = 0.80

    # Normalise columns
    A[0] = A[0] / A[0].sum(axis=0, keepdims=True)

    # B: transition model p(s' | s, a), shape (states, states, actions)
    B = utils.obj_array(1)
    B[0] = np.zeros((n_states[0], n_states[0], n_controls[0]))

    # Under TRACK:
    #   - VISIBLE tends to stay VISIBLE or become OCCLUDED
    #   - OCCLUDED somewhat likely to become VISIBLE, but can drift to GONE
    #   - GONE tends to stay GONE
    #
    # From VISIBLE
    B[0][VisState.VISIBLE,   VisState.VISIBLE,   VisAction.TRACK] = 0.85
    B[0][VisState.OCCLUDED,  VisState.VISIBLE,   VisAction.TRACK] = 0.14
    B[0][VisState.GONE,      VisState.VISIBLE,   VisAction.TRACK] = 0.01
    # From OCCLUDED
    B[0][VisState.VISIBLE,   VisState.OCCLUDED,  VisAction.TRACK] = 0.35
    B[0][VisState.OCCLUDED,  VisState.OCCLUDED,  VisAction.TRACK] = 0.35
    B[0][VisState.GONE,      VisState.OCCLUDED,  VisAction.TRACK] = 0.30
    # From GONE
    B[0][VisState.VISIBLE,   VisState.GONE,      VisAction.TRACK] = 0.05
    B[0][VisState.OCCLUDED,  VisState.GONE,      VisAction.TRACK] = 0.10
    B[0][VisState.GONE,      VisState.GONE,      VisAction.TRACK] = 0.85

    # Under WAIT:
    #   We assume we are keeping position, which can help resolve
    #   OCCLUDED -> VISIBLE, but we also don't want to stay OCCLUDED
    #   forever (reduce "sticky occlusion").
    #
    # From VISIBLE
    B[0][VisState.VISIBLE,   VisState.VISIBLE,   VisAction.WAIT] = 0.90
    B[0][VisState.OCCLUDED,  VisState.VISIBLE,   VisAction.WAIT] = 0.09
    B[0][VisState.GONE,      VisState.VISIBLE,   VisAction.WAIT] = 0.01
    # From OCCLUDED
    B[0][VisState.VISIBLE,   VisState.OCCLUDED,  VisAction.WAIT] = 0.55
    B[0][VisState.OCCLUDED,  VisState.OCCLUDED,  VisAction.WAIT] = 0.30
    B[0][VisState.GONE,      VisState.OCCLUDED,  VisAction.WAIT] = 0.15
    # From GONE
    B[0][VisState.VISIBLE,   VisState.GONE,      VisAction.WAIT] = 0.02
    B[0][VisState.OCCLUDED,  VisState.GONE,      VisAction.WAIT] = 0.08
    B[0][VisState.GONE,      VisState.GONE,      VisAction.WAIT] = 0.90

    # --- Correct normalisation for each action column ---
    for a in range(n_controls[0]):
        col_sums = B[0][:, :, a].sum(axis=0, keepdims=True)
        # Avoid division by zero (should not happen, but safe)
        col_sums[col_sums == 0] = 1.0
        B[0][:, :, a] = B[0][:, :, a] / col_sums

    # C: preferences over observations (we like seeing the target;
    # long-loss is least preferred; recently-lost is mildly preferred
    # over long-lost but not excessively, to reduce WAIT stickiness).
    C = utils.obj_array(1)
    c_vec = np.array([
        0.65,  # SEE_TARGET
        0.25,  # RECENTLY_LOST
        0.10,  # LONG_LOST
    ])
    C[0] = np.log(c_vec / c_vec.sum())

    # D: prior over states – assume initially visible
    D = utils.obj_array(1)
    D[0] = np.array([
        0.8,   # VISIBLE
        0.15,  # OCCLUDED
        0.05,  # GONE
    ])

    agent = Agent(A=A, B=B, C=C, D=D)
    return agent