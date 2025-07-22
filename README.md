# CSC2014 - Digital Image Processing

This project contains solutions for various digital image processing tasks focusing on video analysis and face detection.

## Project Structure



CSC2014/
├── Task A/
│   ├── Section 1.py          # Day/Night detection and brightness adjustment
│   ├── Section 2.py          # Face detection and blurring
│   ├── Section 3.py          # Final output assembly
│   ├── Section 4.py          # Watermarking and final enhancement
├── Task B/
│   └── Task_B.py             # Paragraph extraction from scientific papers
├── Converted Paper (8)/      # Input images for Task B
├── Recorded Videos (4)/      # Input video files
├── Output/
│   ├── Task A/
│   │   ├── Section 1/        # Brightness comparison charts
│   │   ├── Section 2/        # Face-blurred videos
│   │   ├── Section 3/        # Intermediate outputs for final assembly
│   │   └── Final_Output_Videos/ # Watermarked and enhanced videos
│   └── Task B/               # Extracted paragraphs and histogram visualizations
├── Dependency/
│   └── face_detector.xml     # Haar cascade classifier for face detection
└── README.md



## Task A

### Section 1: Day/Night Detection & Brightness Enhancement

*Objective*: Detect whether a video is taken during daytime or nighttime and enhance brightness for nighttime videos.

*Features*:

- Automatic day/night classification based on average brightness and histogram analysis
- Adaptive brightness enhancement for nighttime videos
- Visual comparison charts showing before/after brightness levels
- Support for multiple video formats (.mp4, .avi, .mov, .mkv)

*Algorithm*:

- Analyzes average brightness across video frames
- Uses histogram distribution to determine lighting conditions
- Applies gamma correction and exposure adjustment for nighttime videos

### Section 2: Face Detection & Blurring

*Objective*: Detect and blur all camera-facing faces in videos while maintaining tracking stability.

*Features*:

- *Advanced Face Tracking*: Uses KCF/CSRT trackers with motion prediction
- *Scene Density Analysis*: Automatically adapts to low/medium/high density scenarios
- *Smooth Motion Handling*: Reduces jitter and tracking discontinuities
- *False Positive Filtering*: Multi-layer validation (texture, brightness, aspect ratio)
- *Adaptive Blur Intensity*: Dynamic blur strength based on tracking confidence

*Technical Highlights*:

- Multi-scale face detection for crowded scenes
- Motion prediction for handling occlusion and profile faces
- Bounding box smoothing to eliminate visual artifacts
- Intelligent tracker management with confidence scoring
- Real-time processing with progress monitoring


### Section 3: Final Output Assembly

Combines processed videos and prepares them for final watermarking and output.

### Section 4: Watermarking & Final Enhancement

*Objective*: Overlay a watermark on all output videos and apply final brightness enhancement for improved visual quality.

*Features*:

- Automatic resizing of watermark images to match video resolution
- Adjustable watermark transparency (alpha blending)
- Brightness and contrast enhancement for all frames
- Appends a custom ending video to each output

*Technical Highlights*:

- Uses OpenCV's addWeighted for watermark blending
- Frame-by-frame brightness adjustment using convertScaleAbs
- Supports batch processing of multiple videos


*Output*:

- Watermarked and enhanced videos saved in Output/Task A/Final_Output_Videos/ with prefix watermarked_

---

## Task B: Paragraph Extraction from Scientific Papers

*Objective*: Automatically detect, extract, and sort paragraphs from double/triple-column scientific paper images, filtering out tables and saving results for further analysis.

*Features*:

- Column detection using vertical whitespace analysis
- Paragraph segmentation based on row pixel density and spacing
- Table region filtering to exclude non-paragraph content
- Histogram projection visualizations for both rows and columns
- Batch processing of multiple input images

*Technical Highlights*:

- Uses OpenCV and NumPy for image processing and histogram analysis
- Table regions detected and masked using pixel density thresholds
- Paragraphs sorted top-to-bottom and left-to-right for correct reading order
- Output includes cropped paragraph images and histogram visualizations

*Output*:

- Cropped paragraph images saved in Output/Task B/ with filenames indicating source image, column, and paragraph order
- Row and column histogram images for each input saved in Output/Task B/

*Usage*:

bash
python "Task B/Task_B.py"
