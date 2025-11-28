# POMDP Integration Roadmap
**Sphero Cat Toy - Active Inference Enhancement**

*Based on: sphero_track_test_v2.py (working baseline)*

---

## 📋 Executive Summary

This roadmap outlines the integration of POMDP (Partially Observable Markov Decision Process) decision-making into the working Sphero cat toy system. The goal is to enhance autonomous behavior through **active inference** while maintaining the robust tracking and safety features of v2.

**Target POMDPs:**
1. **Calibration Confidence Monitor** (Primary) - Adaptive system state awareness
2. **Target Behavior Prediction** (Secondary) - Intelligent chase strategies

---

## 🔍 Current System Analysis

### Architecture Overview (sphero_track_test_v2.py)

**Perception Layer:**
- Multi-method ball detection (Hough + YOLO + brightness fusion)
- Kalman filtering for position/velocity estimation
- YOLOv8 segmentation for target detection (cat/dog/person)

**State Estimation:**
- Continuous (x, y) coordinates
- Velocity estimation (vx, vy)
- Smoothed position with exponential filter (ALPHA=0.6)
- NO grid discretization currently

**Control Layer:**
- **Deterministic cascade** (lines 803-856):
  1. Safety mode (boundary/exit prediction) → highest priority
  2. Arrived mode (within TARGET_REACHED_PIX)
  3. Chase mode (distance-based speed profile)
- **Stuck detection** threshold-based escape

**Calibration:**
- Initial automatic calibration (4 cardinal moves)
- Runtime refinement (line 750-762)
- Parameters: `calibration_offset` (heading) and `px_per_speed_per_sec` (scale)

### Existing POMDP Experience

**Previous implementations analyzed:**

1. **sphero_webcam_pomdp.py** (Simple person tracking)
   - Uses pymdp.Agent
   - 3 states: LEFT/CENTER/RIGHT
   - 3 actions: TURN_LEFT/STRAIGHT/TURN_RIGHT
   - Full observability (A = identity matrix)
   - Deterministic transitions
   - **Lesson**: Simple discrete POMDP works, but doesn't leverage uncertainty

2. **run_sphero_bayesian.py** (Circle with collision)
   - 12 heading sectors
   - Collision detection event-driven
   - POMDP mostly ceremonial (not exploiting hidden states)
   - **Lesson**: Don't use POMDP unless there's real uncertainty to reason about

3. **Sphero_track_cat_grid.py** (Grid-based cat tracking)
   - Attempted grid discretization
   - Combined POMDP with vision
   - "Works but needs tweaking" per git log
   - **Lesson**: Grid discretization is viable but needs careful tuning

---

## 🎯 POMDP Opportunities

### Where Uncertainty Exists

With **overhead stationary camera**, we have:
- ✅ Full visibility of play area
- ✅ Direct observation of Sphero and target positions
- ✅ Velocity and trajectory measurements

**But we DON'T directly observe:**
1. **System calibration quality** - Is our heading/scale model drifting?
2. **Target intent/engagement** - Is the person/cat playing or just passing by?
3. **Target future behavior** - Where will they move next?
4. **Occlusion states** - When target briefly disappears (under furniture)
5. **Surface conditions** - Carpet vs hardwood affecting movement

### Decision Points Suitable for POMDP

**Current deterministic decisions that could benefit from probabilistic reasoning:**

| Decision Point | Current (Deterministic) | POMDP Opportunity |
|---------------|------------------------|-------------------|
| System reliability | Assume calibration is good | Infer calibration state, adapt behavior |
| Speed selection | Distance formula | Belief over target engagement → adjust aggression |
| Chase strategy | Direct pursuit | Predict trajectory, choose intercept/orbit/wait |
| Mode selection | Hard priority cascade | Belief over "play session state" |
| Stuck escape | Threshold counter | Infer obstacle type, choose escape strategy |

---

## 🚀 Proposed POMDP Implementations

## POMDP 1: Calibration Confidence Monitor ⭐ PRIMARY

### Motivation
Addresses **real operational problem**: Calibration drift causes:
- Sphero missing targets
- Unexpected boundary approaches
- Increased prediction errors
- System unreliability over time

Currently: Runtime refinement (line 750-762) uses heuristic updates
**POMDP advantage**: Probabilistically infer calibration state, trigger recalibration proactively

### Hidden States (3)

```python
S_CALIB = {
    'WELL_CALIBRATED':  System operating accurately
    'DRIFTING':         Gradual degradation detected
    'BADLY_OFF':        Significant miscalibration
}
```

**Initial belief**: D = [0.8, 0.15, 0.05]  (optimistic start)

### Observations (4)

Derived from existing logged data:

```python
O_CALIB = {
    'LOW_ERROR':      prediction_error < 10px  (line 720)
    'MEDIUM_ERROR':   10px ≤ error < 30px
    'HIGH_ERROR':     30px ≤ error < 50px
    'CRITICAL_ERROR': error ≥ 50px
}
```

**Additional observation signals** (already tracked):
- `abs(angle_diff(actual, expected))` from runtime refinement (line 758)
- `stuck_frames` counter (line 866-868)
- Frequency of safety mode activations

### Actions (4)

```python
A_CALIB = {
    'NORMAL_OPERATION':  # No changes, full performance
        - MAX_SPEED unchanged
        - Standard control loop

    'CAUTIOUS_MODE':     # Reduce risk while uncertain
        - effective_max_speed = MAX_SPEED * 0.7
        - Increase BORDER_MARGIN by 20%
        - More conservative predictions

    'RECALIBRATE':       # Quick 2-direction test
        - Stop current activity
        - Move in 2 cardinal directions (vs 4 in full calib)
        - Update calibration_offset and px_per_speed_per_sec
        - Takes ~5 seconds

    'EMERGENCY_STOP':    # Critical failure
        - Stop Sphero
        - Alert user (LED flash red)
        - Require full recalibration
}
```

### Transition Model B: p(s' | s, a)

**Dynamics**: Calibration degrades over time due to:
- IMU drift
- Surface changes
- Mechanical variations
- Battery discharge

```python
# Example transition probabilities

# Natural drift (NORMAL_OPERATION action):
B[NORMAL_OPERATION] = [
    # s'=WELL   DRIFT   BADLY  | s=
    [  0.95,    0.04,   0.01  ],  # WELL_CALIBRATED
    [  0.05,    0.85,   0.10  ],  # DRIFTING
    [  0.00,    0.05,   0.95  ]   # BADLY_OFF
]

# Recalibration effect (RECALIBRATE action):
B[RECALIBRATE] = [
    # s'=WELL   DRIFT   BADLY  | s=
    [  0.98,    0.02,   0.00  ],  # WELL → very likely fixed
    [  0.85,    0.10,   0.05  ],  # DRIFT → likely fixed
    [  0.60,    0.30,   0.10  ]   # BADLY → maybe fixed
]

# CAUTIOUS_MODE: slower drift
B[CAUTIOUS_MODE] = [
    [  0.97,    0.03,   0.00  ],  # slightly better preservation
    [  0.10,    0.87,   0.03  ],
    [  0.00,    0.10,   0.90  ]
]
```

### Observation Model A: p(o | s)

**Likelihood of observing errors given true calibration state:**

```python
A = [
    # o=LOW   MED    HIGH   CRIT  | s=
    [ 0.70,  0.20,  0.08,  0.02  ],  # WELL_CALIBRATED
    [ 0.20,  0.50,  0.25,  0.05  ],  # DRIFTING
    [ 0.05,  0.15,  0.40,  0.40  ]   # BADLY_OFF
]
```

### Preferences C: p(o)

Prefer low errors:
```python
C = np.log([
    0.6,   # LOW_ERROR (preferred)
    0.25,  # MEDIUM_ERROR (acceptable)
    0.10,  # HIGH_ERROR (avoid)
    0.05   # CRITICAL_ERROR (strongly avoid)
])
```

### Integration Points

**Where to insert** (in main loop, around line 720-730):

```python
# After computing prediction_error
obs_calib = discretize_prediction_error(prediction_error)

# POMDP inference
belief_calib = calibration_agent.infer_states([obs_calib])
calibration_agent.infer_policies()
action_calib = calibration_agent.sample_action()[0]

# Apply action effects
if action_calib == CAUTIOUS_MODE:
    effective_max_speed = int(MAX_SPEED * 0.7)
    effective_border_margin = int(BORDER_MARGIN * 1.2)
elif action_calib == RECALIBRATE:
    perform_quick_recalibration(cap, droid, model, ball_tracker)
    # Reset belief to WELL_CALIBRATED
elif action_calib == EMERGENCY_STOP:
    droid.set_speed(0)
    handle_calibration_failure()
```

**Logging additions:**
- Current belief distribution: `[b_well, b_drift, b_badly]`
- Observation category: `obs_calib`
- Action selected: `action_calib`

---

## POMDP 2: Target Behavior Prediction 🎯 SECONDARY

### Motivation
Enable **anticipatory** rather than reactive chasing:
- Intercept moving targets instead of trailing
- Adjust engagement based on inferred intent
- Maintain optimal "play distance"

### Hidden States (4)

```python
S_TARGET = {
    'STATIONARY':   Target standing/sitting still (low velocity)
    'WANDERING':    Slow movement, no clear direction (path curvature high)
    'DIRECTED':     Walking purposefully (straight trajectory)
    'FLEEING':      Moving away from Sphero (high velocity, away direction)
}
```

### Observations (Continuous → Discrete)

**Raw measurements** (add to tracking):
```python
# Target velocity (not Sphero velocity)
target_history = deque(maxlen=10)  # (goal_x, goal_y, timestamp)
target_vel_x, target_vel_y = compute_target_velocity(target_history)
target_speed = sqrt(target_vel_x^2 + target_vel_y^2)

# Direction relative to Sphero
dx_to_sphero = est_x - goal_x
dy_to_sphero = est_y - goal_y
dot_product = target_vel_x * dx_to_sphero + target_vel_y * dy_to_sphero
moving_toward_sphero = (dot_product > 0)

# Path curvature (acceleration)
target_accel = d(target_velocity)/dt
curvature = |accel|
```

**Discrete observations**:
```python
O_TARGET = {
    'STILL':         target_speed < 5 px/s
    'SLOW_TOWARD':   5-20 px/s, moving_toward_sphero
    'SLOW_AWAY':     5-20 px/s, moving_away_sphero
    'FAST_TOWARD':   >20 px/s, moving_toward
    'FAST_AWAY':     >20 px/s, moving_away
    'ERRATIC':       curvature > threshold (changing direction)
}
```

### Actions (6)

```python
A_TARGET = {
    'DIRECT_CHASE':       Current behavior - aim at current position

    'INTERCEPT':          Predict position T seconds ahead
                          predicted_pos = goal + velocity * T
                          Chase predicted position

    'ORBIT':              Maintain current distance
                          Circle around target at constant radius

    'CAUTIOUS_APPROACH':  Slow advance (if FLEEING detected)
                          Speed = MIN_SPEED, indirect path

    'WAIT_AND_AMBUSH':    Stop moving (if WANDERING)
                          Wait at current position
                          Attract attention by being "prey"

    'RETREAT':            Back away to re-engage
                          Increase distance if too close
}
```

### Transition Model B: p(s' | s, a)

**Natural dynamics** (target behavior evolves):
```python
B[DIRECT_CHASE] = [  # Sphero's actions don't control target directly
    # s'=STAT  WAND  DIR   FLEE  | s=
    [  0.85,  0.10, 0.03, 0.02  ],  # STATIONARY → likely stays
    [  0.20,  0.65, 0.10, 0.05  ],  # WANDERING → might sit or walk
    [  0.05,  0.10, 0.75, 0.10  ],  # DIRECTED → continues walking
    [  0.10,  0.05, 0.25, 0.60  ]   # FLEEING → might keep fleeing
]

# INTERCEPT: aggressive chasing increases FLEEING probability
B[INTERCEPT] = [
    [  0.70,  0.10, 0.10, 0.10  ],
    [  0.15,  0.55, 0.10, 0.20  ],
    [  0.05,  0.10, 0.60, 0.25  ],
    [  0.05,  0.05, 0.15, 0.75  ]   # chasing → more fleeing
]

# WAIT_AND_AMBUSH: stopping might attract wandering target
B[WAIT_AND_AMBUSH] = [
    [  0.90,  0.08, 0.02, 0.00  ],
    [  0.30,  0.60, 0.08, 0.02  ],  # wanderer might approach stopped toy
    [  0.05,  0.15, 0.70, 0.10  ],
    [  0.20,  0.10, 0.10, 0.60  ]
]
```

### Observation Model A: p(o | s)

```python
A = [
    # o=STILL  SLOW_T SLOW_A FAST_T FAST_A ERRAT | s=
    [ 0.80,   0.05,  0.05,  0.02,  0.02,  0.06  ],  # STATIONARY
    [ 0.15,   0.20,  0.20,  0.05,  0.05,  0.35  ],  # WANDERING (erratic)
    [ 0.05,   0.20,  0.20,  0.25,  0.25,  0.05  ],  # DIRECTED (straight)
    [ 0.05,   0.05,  0.35,  0.05,  0.45,  0.05  ]   # FLEEING (fast away)
]
```

### Grid Discretization (Optional Enhancement)

**Hybrid approach** for spatial reasoning:

```python
# Coarse grid overlay (4x4 = 16 zones)
GRID_ROWS = 4
GRID_COLS = 4

def pos_to_grid(x, y, w, h):
    col = min(int(x / (w / GRID_COLS)), GRID_COLS - 1)
    row = min(int(y / (h / GRID_ROWS)), GRID_ROWS - 1)
    return (row, col)

# Extended hidden state: S = (behavior, grid_zone)
# Total states = 4 behaviors × 16 zones = 64 states
# Still tractable for POMDP!
```

**When to use grid:**
- Predicting which zone target will enter next
- Planning patrol routes when no target visible
- Reasoning about occlusion (target in zone behind obstacle)

**When NOT to use grid:**
- Final command execution (use continuous coordinates)
- Speed/heading calculations (keep continuous math)

### Integration Points

**Add after target detection** (around line 740-748):

```python
# Track target history
if target_found:
    target_history.append((goal_x, goal_y, time.time()))

# Compute target kinematics
target_vel_x, target_vel_y = compute_target_velocity(target_history)
target_speed = compute_speed(target_vel_x, target_vel_y)
toward_sphero = check_direction_toward_sphero(goal_x, goal_y,
                                                target_vel_x, target_vel_y,
                                                est_x, est_y)

# Discretize observation
obs_target = discretize_target_behavior(target_speed, toward_sphero, curvature)

# POMDP inference
belief_target = target_agent.infer_states([obs_target])
target_agent.infer_policies()
action_target = target_agent.sample_action()[0]

# Modify chase behavior based on action
if action_target == INTERCEPT:
    T_PREDICT = 0.5  # seconds
    predicted_goal_x = goal_x + target_vel_x * T_PREDICT
    predicted_goal_y = goal_y + target_vel_y * T_PREDICT
    goal_x, goal_y = predicted_goal_x, predicted_goal_y  # override

elif action_target == ORBIT:
    # Calculate tangent to maintain distance
    angle_to_target = get_screen_angle((est_x, est_y), (goal_x, goal_y))
    orbit_angle = (angle_to_target + 90) % 360  # perpendicular
    # Keep current distance, move tangentially

elif action_target == WAIT_AND_AMBUSH:
    should_move = False  # override movement
```

---

## 📐 Implementation Plan

### Phase 1: Foundation (Week 1)

**Goal**: Set up POMDP infrastructure without disrupting working system

**Tasks**:
1. ✅ Read and analyze existing code
2. ✅ Review pymdp usage patterns from previous implementations
3. Create POMDP utility module (`sphero_pomdp_utils.py`)
   - Observation discretization functions
   - Transition/observation model builders
   - Logging helpers
4. Add data collection to v2:
   - Target velocity tracking (history buffer)
   - Enhanced error metrics logging
5. Test data collection without POMDP (validate measurements)

**Deliverables**:
- `sphero_pomdp_utils.py` with pymdp helpers
- Modified `sphero_track_test_v2.py` with enhanced logging
- Validation: Run system, verify new data fields in CSV

---

### Phase 2: Calibration POMDP (Week 2)

**Goal**: Implement and test calibration confidence monitor

**Tasks**:
1. Build calibration POMDP model:
   ```python
   def build_calibration_pomdp():
       A = build_observation_model_calib()
       B = build_transition_model_calib()
       C = build_preferences_calib()
       D = build_prior_calib()
       return Agent(A=A, B=B, C=C, D=D)
   ```
2. Integrate into main loop (non-disruptive):
   - Run POMDP inference in parallel
   - Log beliefs and actions WITHOUT executing them yet
3. Offline analysis:
   - Plot belief evolution over time
   - Correlate with actual prediction errors
   - Tune transition/observation models
4. Enable action execution:
   - Start with CAUTIOUS_MODE only
   - Add RECALIBRATE once stable
5. Testing scenarios:
   - Fresh calibration → should stay WELL_CALIBRATED
   - Manually move Sphero (simulate drift) → should detect DRIFTING
   - Run for extended period → observe natural drift detection

**Deliverables**:
- `sphero_track_test_v3_calib_pomdp.py`
- Calibration POMDP belief logs
- Analysis notebook/script showing belief accuracy

---

### Phase 3: Target Behavior POMDP (Week 3-4)

**Goal**: Add intelligent chase strategies

**Tasks**:
1. Implement target kinematics tracking:
   - Velocity estimation from history
   - Direction classification (toward/away)
   - Path curvature calculation
2. Build target behavior POMDP:
   ```python
   def build_target_pomdp():
       A = build_observation_model_target()
       B = build_transition_model_target()
       C = build_preferences_target()
       D = build_prior_target()
       return Agent(A=A, B=B, C=C, D=D)
   ```
3. Implement action execution:
   - INTERCEPT: prediction math
   - ORBIT: tangent calculation
   - WAIT_AND_AMBUSH: stop logic
4. Test with human volunteer:
   - Walk straight line → should INTERCEPT
   - Walk in circles → should ORBIT or INTERCEPT arc
   - Stand still → should CAUTIOUS_APPROACH or WAIT
   - Run away → should CAUTIOUS_APPROACH (not aggressive chase)
5. Tune models based on observed behavior

**Deliverables**:
- Full dual-POMDP system
- Video demonstrations of different chase behaviors
- Performance comparison: v2 (direct chase) vs v3 (POMDP)

---

### Phase 4: Grid Enhancement (Optional)

**Goal**: Add spatial reasoning if needed

**Tasks**:
1. Implement hybrid coordinate system
2. Add grid-based state: S = (behavior, zone)
3. Test occlusion handling:
   - Hide behind obstacle
   - POMDP should maintain belief about zone
4. Patrol behavior when no target visible

**Deliverables**:
- Grid-enhanced POMDP
- Occlusion handling demonstrations

---

## 🧪 Testing Strategy

### Unit Tests
- Observation discretization correctness
- Model probability distributions (sum to 1, valid)
- Action execution (intercept math, orbit geometry)

### Integration Tests
- POMDP runs without crashes
- Beliefs update correctly given observations
- Actions modify behavior as expected

### Behavioral Tests
**Calibration POMDP:**
- Detects gradual drift
- Triggers recalibration appropriately
- Doesn't false-alarm on normal operation

**Target POMDP:**
- Correctly classifies target behaviors
- Chooses sensible actions
- Improves engagement compared to baseline

### Performance Metrics
- **Calibration accuracy**: prediction_error over time
- **Target engagement**: time spent within optimal distance
- **Computational overhead**: loop frequency maintained at ~20Hz
- **Robustness**: handles edge cases without crashes

---

## 📊 Evaluation Criteria

### Success Metrics

**Calibration POMDP:**
- [ ] Belief correlates with actual calibration state (>80% accuracy)
- [ ] Reduces calibration-related failures by 50%
- [ ] Triggers recalibration before catastrophic drift
- [ ] Maintains <5ms computational overhead per loop

**Target POMDP:**
- [ ] Intercept actions reduce time-to-target by 20%
- [ ] Orbit maintains stable distance (std dev < 30px)
- [ ] Correctly identifies FLEEING state (>70% accuracy)
- [ ] Human testers report more "intelligent" behavior

### Failure Criteria (Abort if...)
- POMDP inference takes >50ms (unacceptable latency)
- Beliefs oscillate chaotically (model poorly tuned)
- Actions make behavior worse than deterministic baseline
- System becomes less stable/robust

---

## 🔧 Technical Considerations

### PyMDP Integration

**Based on existing usage** (sphero_webcam_pomdp.py, run_sphero_bayesian.py):

```python
from pymdp import utils
from pymdp.agent import Agent

# Standard pattern:
num_obs = [N_OBSERVATIONS]
num_states = [N_STATES]
num_controls = [N_ACTIONS]

A = utils.obj_array(1)  # observation model
B = utils.obj_array(1)  # transition model
C = utils.obj_array_uniform(num_obs)  # preferences
D = utils.obj_array(1)  # prior

agent = Agent(A=A, B=B, C=C, D=D)

# In loop:
belief = agent.infer_states([observation])
agent.infer_policies()
action = agent.sample_action()
```

### Computational Budget

**Current loop**: ~0.05s = 50ms at 20Hz

**POMDP overhead budget**: <10ms total
- Calibration POMDP: ~2-3ms (small state space)
- Target POMDP: ~5-7ms (larger state space)

**Optimization if needed:**
- Cache policy computations
- Reduce policy horizon
- Use simplified inference (MAP instead of full belief)

### Logging Schema

**Enhanced CSV columns**:
```python
# Existing columns +
"calib_belief_well",
"calib_belief_drift",
"calib_belief_badly",
"calib_obs",
"calib_action",
"target_belief_stationary",
"target_belief_wandering",
"target_belief_directed",
"target_belief_fleeing",
"target_obs",
"target_action",
"target_vel_x",
"target_vel_y",
"target_speed",
"target_curvature"
```

---

## 📚 References

### Papers (sources/ folder)

1. **Active Inference and Robot Control** (pio-lopez-et-al-2016)
   - Theoretical foundation for using active inference in robotics
   - Relevant sections: Sections 2-3 (generative models, free energy)

2. **Why Learn if You Can Infer?** (23_Why_learn_if_you_can_infer_)
   - Comparison of learning vs inference approaches
   - Relevant: When POMDP is preferable to RL

3. **Entropy paper** (entropy-24-00469-v2.pdf)
   - Uncertainty quantification
   - Relevant: Observation model design

4. **Additional sources** (2206.10313v1.pdf, s00422-018-0753-2.pdf)
   - Implementation details and case studies

### Code References

- `sphero_webcam_pomdp.py`: Simple 3-state POMDP template
- `run_sphero_bayesian.py`: Event-driven POMDP integration
- `sphero_track_test_v2.py`: Working baseline (DO NOT BREAK!)

### External Resources

- PyMDP documentation: https://github.com/infer-actively/pymdp
- Active Inference tutorials: https://github.com/infer-actively/pymdp-tutorial

---

## ⚠️ Risk Mitigation

### Risk 1: POMDP Makes System Worse
**Mitigation**:
- Keep v2 as baseline
- A/B testing framework
- Easy rollback (POMDP toggle flag)

### Risk 2: Computational Overhead
**Mitigation**:
- Profile early
- Simplify state spaces
- Offload to background thread if needed

### Risk 3: Model Misspecification
**Mitigation**:
- Start with conservative priors
- Online parameter tuning
- Extensive logging for debugging

### Risk 4: Over-Engineering
**Mitigation**:
- Implement incrementally
- Validate each phase independently
- Only add grid if simple version insufficient

---

## 🎯 Next Steps

1. **Review this roadmap** - Approve/modify plan
2. **Set up development branch**: `git checkout -b pomdp-integration`
3. **Phase 1 - Week 1**: Create `sphero_pomdp_utils.py`
4. **Daily testing**: Run on real hardware, iterate quickly
5. **Document learnings**: Update roadmap with findings

---

## 📝 Open Questions

1. **Grid granularity**: 4x4, 5x5, or continuous only?
   - *Recommend*: Start continuous, add grid in Phase 4 if needed

2. **Preferences C**: How strongly should we prefer low errors?
   - *Recommend*: Start mild, increase if POMDP too passive

3. **Action frequency**: POMDP every frame or every N frames?
   - *Recommend*: Every frame for calibration, every 3-5 frames for target

4. **Hierarchical vs Flat**: Should calibration monitor override target POMDP?
   - *Recommend*: Yes - safety/calibration has priority

5. **Real cat testing timeline**: When can we test with actual cat?
   - *Answer*: After Phase 3 completes successfully with human

---

## 📅 Timeline Summary

| Phase | Duration | Deliverable | Status |
|-------|----------|-------------|--------|
| Phase 1: Foundation | Week 1 | POMDP utils + enhanced logging | 🔲 Not started |
| Phase 2: Calibration POMDP | Week 2 | Working calibration monitor | 🔲 Not started |
| Phase 3: Target POMDP | Week 3-4 | Intelligent chase behaviors | 🔲 Not started |
| Phase 4: Grid Enhancement | Optional | Spatial reasoning | 🔲 Optional |

**Total estimated time**: 3-4 weeks with incremental testing

---

**Document Version**: 1.0
**Last Updated**: 2025-11-28
**Author**: Based on analysis of sphero_track_test_v2.py and pymdp implementations
