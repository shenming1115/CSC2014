# CSC2014 - Digital Image Processing

This project contains solutions for various digital image processing tasks focusing on video analysis and face detection.

## Project Structure

```
CSC2014/
├── Task A/
│   ├── Section 1.py          # Day/Night detection and brightness adjustment
│   └── Section 2.py          # Face detection and blurring
├── Recorded Videos (4)/      # Input video files
├── Output/
│   └── Task A/
│       ├── Section 1/        # Brightness comparison charts
│       └── Section 2/        # Face-blurred videos
├── Dependency/
│   └── face_detector.xml     # Haar cascade classifier for face detection
└── README.md
```

## Task A

### Section 1: Day/Night Detection & Brightness Enhancement

**Objective**: Detect whether a video is taken during daytime or nighttime and enhance brightness for nighttime videos.

**Features**:

- Automatic day/night classification based on average brightness and histogram analysis
- Adaptive brightness enhancement for nighttime videos
- Visual comparison charts showing before/after brightness levels
- Support for multiple video formats (.mp4, .avi, .mov, .mkv)

**Algorithm**:

- Analyzes average brightness across video frames
- Uses histogram distribution to determine lighting conditions
- Applies gamma correction and exposure adjustment for nighttime videos

### Section 2: Face Detection & Blurring

**Objective**: Detect and blur all camera-facing faces in videos while maintaining tracking stability.

**Features**:

- **Advanced Face Tracking**: Uses KCF/CSRT trackers with motion prediction
- **Scene Density Analysis**: Automatically adapts to low/medium/high density scenarios
- **Smooth Motion Handling**: Reduces jitter and tracking discontinuities
- **False Positive Filtering**: Multi-layer validation (texture, brightness, aspect ratio)
- **Adaptive Blur Intensity**: Dynamic blur strength based on tracking confidence

**Technical Highlights**:

- Multi-scale face detection for crowded scenes
- Motion prediction for handling occlusion and profile faces
- Bounding box smoothing to eliminate visual artifacts
- Intelligent tracker management with confidence scoring
- Real-time processing with progress monitoring

### Section 3: TBC

*To be completed*

### Section 4: TBC

*To be completed*

## Dependencies

- OpenCV (opencv-python, opencv-contrib-python)
- NumPy
- Python 3.7+

## Usage

### Running Section 1 (Day/Night Detection):

```bash
python "Task A/Section 1.py"
```

### Running Section 2 (Face Blurring):

```bash
python "Task A/Section 2.py"
```

## Input/Output

- **Input**: Place video files in `Recorded Videos (4)/` directory
- **Output**:
  - Section 1: Brightness analysis charts in `Output/Task A/Section 1/`
  - Section 2: Blurred videos with prefix "blurred_" in `Output/Task A/Section 2/`

## Notes

- Face cascade classifier (`face_detector.xml`) must be present in `Dependency/` folder
- Output directories are created automatically
- Progress is displayed every 100 processed frames
- Supports various video codecs with automatic format detection
