"""
Sphero BOLT: small-circle pattern with collision avoidance,
driven by a minimal POMDP (using pymdp).

Behavior:
- Robot drives in a tight circle (small radius).
- When it collides with something, it:
    * turns red,
    * stops briefly,
    * re-aims ~90 degrees,
    * resumes circling (green LED).

Requirements:
    pip install spherov2 bleak pymdp

Make sure your Sphero BOLT is awake and near the computer.
"""

import sys
sys.coinit_flags = 0  # 0 = COINIT_MULTITHREADED (MTA); must be FIRST

import time
import numpy as np

from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI, EventType
from spherov2.types import Color

from pymdp import utils
from pymdp.agent import Agent


# === POMDP + control configuration ===

# Number of discrete heading sectors.
# 12 => 30° increments => tight circle.
N_HEADINGS = 12

# Sphero movement parameters tuned for small circles
SPHERO_SPEED = 40        # 0–255
STEP_DELAY = 0.05        # seconds between heading updates
NUM_STEPS = 600          # how long to run (iterations)


def build_pomdp_agent():
    """
    Build a minimal POMDP generative model and return a pymdp Agent.

    Model:
        - 1 observation modality
        - 1 hidden-state factor ("heading sector")
        - 1 control factor (which sector to move to)
        - Full observability: o = s (A = identity)
        - Deterministic transitions: next_state = action_index
        - Uniform priors and preferences (D and C)
    """

    # Dimensions
    num_obs = [N_HEADINGS]       # one modality, N possible observations
    num_states = [N_HEADINGS]    # one hidden-state factor, N states
    num_controls = [N_HEADINGS]  # one control factor, N possible actions

    # --- A: observation likelihood p(o | s)
    # Full observability: A is identity
    A = utils.obj_array(1)
    A[0] = np.eye(N_HEADINGS)

    # --- B: state transitions p(s' | s, u)
    # Define next_state = action_index, independent of previous state.
    B = utils.obj_array(1)
    B[0] = np.zeros((N_HEADINGS, N_HEADINGS, N_HEADINGS))

    for s in range(N_HEADINGS):
        for a in range(N_HEADINGS):
            B[0][a, s, a] = 1.0

    # --- C: preferences over observations (log-uniform)
    C = utils.obj_array_uniform(num_obs)

    # --- D: prior over initial states (uniform)
    D = utils.obj_array(1)
    D[0] = np.ones(N_HEADINGS) / N_HEADINGS

    agent = Agent(A=A, B=B, C=C, D=D)
    return agent


def sector_to_heading_deg(sector_index: int) -> int:
    """
    Map a discrete sector index (0..N_HEADINGS-1) to a physical heading in degrees.
    """
    step = 360.0 / N_HEADINGS
    return int((sector_index * step) % 360)


def run_pomdp_circle(droid: SpheroEduAPI):
    """
    Run POMDP-controlled small-circle behavior with collision avoidance.

    - POMDP tracks a discrete heading sector.
    - We enforce a deterministic circle policy: next_sector = (current+1) % N.
    - Collision event sets a flag; main loop re-aims ~90° and keeps circling.
    """

    agent = build_pomdp_agent()

    # Shared state for collision flag (modified by event callback)
    collision_state = {"hit": False}

    def on_collision(api: SpheroEduAPI):
        """Callback executed in a separate thread when a collision is detected."""
        print("Collision event detected!")
        collision_state["hit"] = True

    # Register collision event
    droid.register_event(EventType.on_collision, on_collision)

    # Optional: reset aim so 0° corresponds to current heading
    droid.reset_aim()

    # Initial state
    current_sector = 0
    observation = [current_sector]

    print("Starting POMDP-driven small circle with collision avoidance...")
    droid.set_main_led(Color(0, 255, 0))  # green

    try:
        for t in range(NUM_STEPS):
            # 1) State inference given observation
            qs = agent.infer_states(observation)

            # 2) Policy evaluation (we won't use the sampled action yet,
            #    but the agent is still doing active inference over policies)
            agent.infer_policies()

            # 3) Handle collision if it happened
            if collision_state["hit"]:
                print("Handling collision in main loop...")
                # Stop and signal red
                droid.set_speed(0)
                droid.set_main_led(Color(255, 0, 0))
                time.sleep(0.3)

                # Re-aim ~90 degrees by shifting the sector
                turn_quarter = N_HEADINGS // 4  # e.g. 12 -> 3 sectors -> 90°
                current_sector = (current_sector + turn_quarter) % N_HEADINGS

                # Clear collision flag
                collision_state["hit"] = False

                # Back to green for normal circling
                droid.set_main_led(Color(0, 255, 0))

            # 4) Deterministic circle policy: advance one sector
            next_sector = (current_sector + 1) % N_HEADINGS

            # 5) Convert sector to physical heading and send command
            heading_deg = sector_to_heading_deg(next_sector)
            droid.set_heading(heading_deg)
            droid.set_speed(SPHERO_SPEED)

            # 6) Environment "transition" & new observation
            current_sector = next_sector
            observation = [current_sector]   # full observability: o = s

            # 7) Wait before next step
            time.sleep(STEP_DELAY)

        print("Finished POMDP circle run.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        droid.set_speed(0)
        droid.set_main_led(Color(255, 255, 255))
        time.sleep(0.2)


def main():
    print("Searching for Sphero toy...")
    toy = scanner.find_toy()

    if not toy:
        print("No Sphero toy found. Make sure it is awake and nearby.")
        return

    print(f"Found {toy.name}. Connecting...")
    with SpheroEduAPI(toy) as droid:
        print("Connected to Sphero.")
        droid.set_main_led(Color(0, 0, 255))  # blue for "connected"
        time.sleep(0.5)

        run_pomdp_circle(droid)

    print("Disconnected from Sphero.")


if __name__ == "__main__":
    main()