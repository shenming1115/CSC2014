import cv2
import numpy as np
import os
# Section 1: Video Brightness Enhancement
# This script detects whether a video is taken at night and automatically increases brightness if needed.

def is_night_frame(frame, brightness_threshold=80):
    # Check if the current frame is a nighttime scene: convert to grayscale, calculate average brightness, and compare to threshold.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness < brightness_threshold, avg_brightness

def increase_brightness(frame, value=50):
    # Increase frame brightness: convert to HSV color space and adjust only the V (brightness) channel for better color preservation.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def process_video(input_path, output_path, brightness_increase=50, brightness_threshold=80):
    # Process video: read frame by frame, check for nighttime, and increase brightness if needed. Save processed video to output path.
    # Also calculate and print average brightness before and after processing.
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        # If the video file cannot be opened, print error message.
        print(f"❌ Unable to open video file: {input_path}")
        return None, None

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get video width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Get video height
    fps    = int(cap.get(cv2.CAP_PROP_FPS))          # Get video frame rate

    out = cv2.VideoWriter(output_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, (width, height))

    total_before = 0  # Sum of brightness before processing for all frames
    total_after = 0   # Sum of brightness after processing for all frames
    frame_count = 0   # Frame counter
    is_night = False  # Flag for nighttime video

    while True:
        ret, frame = cap.read()  # Read one frame
        if not ret:
            break  # Exit loop if no more frames
        night, brightness = is_night_frame(frame, brightness_threshold)  # Check if current frame is nighttime
        is_night = is_night or night  # If any frame is nighttime, treat whole video as nighttime
        total_before += brightness    # Add brightness before processing
        frame_count += 1             # Increment frame count

        if is_night:
            frame = increase_brightness(frame, brightness_increase)  # Increase brightness for nighttime frames
            bright_after = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))  # Calculate brightness after enhancement
        else:
            bright_after = brightness  # Keep original brightness for daytime frames

        total_after += bright_after   # Add brightness after processing
        out.write(frame)              # Write frame to output video

    cap.release()   # Release video reading resources
    out.release()   # Release video writing resources

    avg_before = total_before / frame_count  # Calculate average brightness before processing
    avg_after = total_after / frame_count    # Calculate average brightness after processing
    print(f"✅ Processing completed: {input_path} (Average brightness {avg_before:.1f} → {avg_after:.1f})")
    return avg_before, avg_after

# ==== Main Processing Section ====
# Define the list of input video files to process. Each video will be checked for nighttime and enhanced if needed.
video_files = [
    r"c:\Users\User\Desktop\CS Y2S1\CSC2014_FinalAssignment\CSC2014\Recorded Videos (4)\alley.mp4",      # Alley video
    r"c:\Users\User\Desktop\CS Y2S1\CSC2014_FinalAssignment\CSC2014\Recorded Videos (4)\office.mp4",     # Office video
    r"c:\Users\User\Desktop\CS Y2S1\CSC2014_FinalAssignment\CSC2014\Recorded Videos (4)\singapore.mp4",  # Singapore street
    r"c:\Users\User\Desktop\CS Y2S1\CSC2014_FinalAssignment\CSC2014\Recorded Videos (4)\traffic.mp4"     # Traffic intersection
]
brightness_before = []  # Store average brightness before processing for each video
brightness_after = []   # Store average brightness after processing for each video

for video in video_files:
    # Process each video, output the enhanced result, and record brightness statistics.
    print(f"\n🎬 Processing video: {video}")
    base = os.path.splitext(os.path.basename(video))[0]  # Get video filename (without extension)
    output_path = os.path.join("Output", "Task A", "Section 1", f"brightness_{base}.mp4")  # Build output path
    before, after = process_video(video, output_path)  # Call processing function
    brightness_before.append(before if before else 0)  # Record brightness before processing
    brightness_after.append(after if after else 0)     # Record brightness after processing

