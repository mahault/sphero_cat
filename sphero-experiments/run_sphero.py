import time
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color

# Configuration
CIRCLE_SPEED = 40          # slower = tighter + more controlled
HEADING_INCREMENT = 30     # big turn = small circle
STEP_DELAY = 0.05          # smoother updates

class SpheroCircleBot:
    def __init__(self, sphero: SpheroEduAPI):
        # sphero is an already-connected SpheroEduAPI instance
        self.sphero = sphero
        self.current_heading = 0
        self.collision_detected = False  # Placeholder for future use
        self.running = True

    def avoid_obstacle(self):
        """Navigate around an obstacle (placeholder – collision not wired yet)."""
        print("Avoiding obstacle...")

        # Stop
        self.sphero.set_speed(0)
        time.sleep(0.3)

        # Show red LED
        self.sphero.set_main_led(Color(255, 0, 0))

        # Back up
        print("Backing up...")
        reverse_heading = (self.current_heading + 180) % 360
        self.sphero.set_heading(reverse_heading)
        self.sphero.set_speed(80)
        time.sleep(1.5)

        # Turn 90 degrees to go around obstacle
        print("Turning to avoid...")
        self.current_heading = (self.current_heading + 90) % 360
        self.sphero.set_speed(0)
        time.sleep(0.3)

        # Move forward to clear obstacle
        print("Moving around obstacle...")
        self.sphero.set_heading(self.current_heading)
        self.sphero.set_speed(100)
        time.sleep(2.0)

        # Turn back a bit to resume circle
        self.current_heading = (self.current_heading - 45) % 360

        # Resume circle indicator (green LED)
        self.sphero.set_main_led(Color(0, 255, 0))
        print("Resuming circle pattern...")

        # Reset collision flag
        self.collision_detected = False
        time.sleep(0.5)

    def drive_in_circles(self):
        """Main loop: drive in circles."""
        print("Starting circle pattern...")
        self.sphero.set_main_led(Color(0, 255, 0))  # Green

        while self.running:
            try:
                if self.collision_detected:
                    self.avoid_obstacle()
                    continue

                # Increment heading to create circular motion
                self.current_heading = (self.current_heading + HEADING_INCREMENT) % 360

                # Set heading and speed
                self.sphero.set_heading(self.current_heading)
                self.sphero.set_speed(CIRCLE_SPEED)

                # Small delay between updates (controls circle smoothness)
                time.sleep(STEP_DELAY)

            except KeyboardInterrupt:
                print("\nInterrupted by user")
                self.running = False
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                self.running = False
                break

    def stop(self):
        """Stop the robot (does not close context; 'with' handles that)."""
        print("\nStopping...")
        try:
            self.running = False
            self.sphero.set_speed(0)
            self.sphero.set_main_led(Color(255, 255, 255))  # White
            time.sleep(0.3)
        except Exception as e:
            print(f"Error during stop: {e}")

    def run(self):
        """Wrapper to start and safely stop."""
        try:
            print("Running in circles")
            self.drive_in_circles()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()


def main():
    print("Searching for Sphero toy...")
    toy = scanner.find_toy()
    if not toy:
        print("No Sphero toy found.")
        return

    print(f"Found {toy.name}. Connecting...")
    # Use context manager to connect and disconnect cleanly
    with SpheroEduAPI(toy) as droid:
        print("Connected. Starting circle bot.")
        # You can still test LED here:
        droid.set_main_led(Color(0, 0, 255))  # Blue
        time.sleep(0.5)

        bot = SpheroCircleBot(droid)
        bot.run()

    print("Disconnected.")


if __name__ == "__main__":
    main()