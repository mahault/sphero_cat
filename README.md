# Sphero Cat Toy

An autonomous cat toy using a Sphero robot ball that tracks and chases cats, dogs, or people using computer vision and YOLO object detection.

## 🎯 Current Working Script: `sphero_track_test_v2.py`

This is the production-ready version featuring robust ball detection, automatic calibration, and intelligent navigation with safety features.

### What It Does

The script creates an autonomous cat toy that:

1. **Detects the Sphero ball** using a multi-method fusion approach:
   - Hough circle detection for geometric precision
   - YOLO object detection (sports ball class) for robust recognition
   - Brightness-based blob detection for LED tracking
   - Kalman filtering for smooth position tracking and velocity estimation

2. **Tracks targets** using YOLOv8 segmentation:
   - Primary targets: Cats and dogs
   - Secondary targets: People (when no pets are detected)
   - Returns to center when no targets are found

3. **Navigates intelligently**:
   - Automatic calibration for heading orientation and movement scale
   - Real-time calibration refinement during operation
   - Predictive boundary avoidance with velocity-aware escape routes
   - Stuck detection and escape behavior
   - Distance-based speed control (slows down as it approaches target)
   - Velocity-aware braking to prevent overshooting

4. **Safety features**:
   - Border margin enforcement to keep Sphero in the play area
   - Predictive exit detection based on current velocity
   - Automatic return to center when near boundaries
   - Stuck detection with automatic escape to center

5. **Comprehensive logging**:
   - Logs all state data to CSV files with timestamps
   - Records position, velocity, commands, errors, and system states
   - Useful for debugging and performance tuning

### How to Run

#### Prerequisites

```bash
pip install spherov2 opencv-python numpy ultralytics
```

You'll also need:
- A Sphero robot (BOLT, SPRK+, or compatible model)
- A webcam with a clear overhead view of the play area
- YOLOv8 model weights (`yolov8s-seg.pt` - will auto-download on first run)

#### Setup

1. **Position your camera**: Mount it overhead with a clear view of the play area
2. **Set camera index**: Edit `CAM_INDEX` in the script if your camera isn't at index 1
3. **Power on your Sphero**: Make sure it's charged and in range

#### Running the Script

```bash
python sphero_track_test_v2.py
```

The script will:
1. Scan for and connect to your Sphero
2. Open the camera feed
3. Set Sphero's LED to white for better tracking
4. Show an **alignment screen** with a green arrow pointing up
   - Physically rotate your Sphero so its forward direction (the LED aim point) aligns with "up" in the camera view
   - Press **SPACE** or **ENTER** when aligned
5. Run **automatic calibration**:
   - The Sphero will move in 4 cardinal directions
   - The system will calculate heading offset and movement scale
   - This takes about 10-15 seconds
6. Start the main tracking loop

#### Controls

- **Q**: Quit the application
- The Sphero will automatically chase detected cats, dogs, or people
- Press **Ctrl+C** to emergency stop

#### Configuration

Key parameters you can adjust at the top of the script:

```python
CAM_INDEX = 1                # Your camera device index
MAX_SPEED = 30               # Maximum Sphero speed (0-255)
MIN_SPEED = 5                # Minimum movement speed
TARGET_REACHED_PIX = 35      # How close to get to target (pixels)
BORDER_MARGIN = 100          # Safety margin from edges (pixels)
```

### Output

The script creates timestamped CSV log files:
```
sphero_log_2025-11-28_14-30-45.csv
```

Each log contains:
- Timestamp and loop index
- Estimated and measured ball positions
- Goal position and target type
- Calibration values
- Command heading and speed
- Distances, errors, and system states
- Velocity measurements

## 📁 Project Evolution

### Early Versions

**`run_sphero.py`**
- Basic Sphero control
- Simple circular motion patterns
- Obstacle avoidance placeholder (no vision)
- Foundation for motor control

**`run_sphero_grid.py`** & **`run_sphero_track.py`**
- Early experiments with waypoint navigation
- Grid-based movement patterns
- No computer vision integration

### POMDP and Bayesian Approaches

**`run_sphero_bayesian.py`**
- Attempted Bayesian position estimation
- Probabilistic tracking experiments
- Explored uncertainty handling

**`sphero_webcam_pomdp.py`**
- POMDP (Partially Observable Markov Decision Process) framework
- Used pymdp for decision-making under uncertainty
- Complex state estimation
- *Abandoned due to complexity and performance issues*

### Cat Tracking Iterations

**`sphero_track_cat.py`**
- First version with actual cat tracking using YOLO
- Basic POMDP integration
- Combined vision with decision theory
- Foundation for autonomous behavior

**`Sphero_track_cat_grid.py`**
- Added grid-based navigation to cat tracking
- Quantized movement space
- Hybrid approach between POMDP and direct control
- Current git HEAD shows this "works but needs tweaking"

### Test Versions (Leading to v2)

**`spher_tratck_test.py`** (Selected in editor)
- Robust multi-method ball detection (Hough + YOLO + brightness)
- Kalman filtering for ball tracking
- Introduced calibration system
- Early safety features
- Uses vision latency compensation

**`sphero_track_test.py`**
- Similar to above but intermediate iteration
- Testing different parameter sets

**`sphero_track_test_v2.py`** ✅ **WORKING VERSION**
- Removed vision latency extrapolation (caused instability)
- Simplified to measurement smoothing with Kalman filtering
- Improved calibration (both auto and runtime refinement)
- Better safety system with predictive boundary checking
- Distance-first speed profile with velocity-aware braking
- Stuck detection and escape behavior
- Comprehensive logging
- **Most stable and reliable version**

## 🔧 Key Technical Improvements in v2

1. **Simplified State Estimation**: Removed latency compensation extrapolation that caused oscillations; now uses simple exponential smoothing with Kalman position estimates

2. **Robust Ball Detection**: Fuses three detection methods (Hough circles, YOLO sports ball, brightness blobs) with confidence scoring

3. **Automatic Calibration**: Moves in cardinal directions to self-calibrate heading offset and movement scale

4. **Predictive Safety**: Uses commanded velocity model to predict boundary exits before they happen

5. **Intelligent Speed Control**: Distance-based speed ramping with velocity-aware braking prevents overshooting

6. **Runtime Adaptation**: Continuously refines calibration offset during operation based on observed movement

## 🐱 Tips for Best Results

1. **Lighting**: Ensure good, even lighting so the white Sphero LED is visible
2. **Contrast**: Use a dark floor for better ball detection
3. **Camera Height**: Mount camera 1.5-2 meters high for best overhead view
4. **Play Area**: Clear, flat space about 2x2 meters minimum
5. **Calibration**: Do the initial alignment carefully for best tracking
6. **Speed Tuning**: If Sphero is too aggressive, reduce `MAX_SPEED`; if too timid, increase it

## 📊 Troubleshooting

**Sphero not found:**
- Check Bluetooth is enabled
- Make sure Sphero is charged and awake
- Try moving it or putting it in the charger briefly to wake it

**Ball not detected:**
- Increase brightness in the room
- Adjust `BRIGHTNESS_THRESHOLD` (lower = more sensitive)
- Check that Sphero LED is set to white
- Ensure camera has clear view with no obstructions

**Poor tracking:**
- Re-run calibration with better alignment
- Adjust `BORDER_MARGIN` if Sphero escapes the view
- Check camera is stable and not moving
- Reduce `MAX_SPEED` for more controlled movement

**Sphero goes off-screen:**
- Increase `BORDER_MARGIN`
- Decrease `PREDICT_HORIZON_SEC` for earlier boundary detection
- Ensure camera covers the full play area

## 📝 License

This is a personal project for cat entertainment and robotics learning.

## 🙏 Acknowledgments

- Uses [spherov2](https://github.com/artificial-intelligence-class/spherov2.py) for Sphero control
- Uses [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- OpenCV for computer vision processing
