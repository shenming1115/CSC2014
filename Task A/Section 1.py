import cv2
import numpy as np
import matplotlib.pyplot as plt

def is_night_frame(frame, brightness_threshold=80):
    """Determine if a frame represents nighttime based on brightness threshold"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness < brightness_threshold, avg_brightness

def increase_brightness(frame, value=50):
    """Increase brightness using HSV color space for better color preservation"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def process_video(input_path, output_path, brightness_increase=50, brightness_threshold=80):
    """Process video: detect nighttime scenes, adjust brightness, and output enhanced video"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Unable to open video file: {input_path}")
        return None, None

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    total_before = 0
    total_after = 0
    frame_count = 0
    is_night = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        night, brightness = is_night_frame(frame, brightness_threshold)
        is_night = is_night or night
        total_before += brightness
        frame_count += 1

        if is_night:
            frame = increase_brightness(frame, brightness_increase)
            bright_after = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            bright_after = brightness

        total_after += bright_after
        out.write(frame)

    cap.release()
    out.release()

    avg_before = total_before / frame_count
    avg_after = total_after / frame_count
    print(f"✅ Processing completed: {input_path} (Average brightness {avg_before:.1f} → {avg_after:.1f})")
    return avg_before, avg_after

# ==== Main Processing Section ====
video_files = ["Recorded Videos (4)/alley.mp4", "Recorded Videos (4)/office.mp4", "Recorded Videos (4)/singapore.mp4", "Recorded Videos (4)/traffic.mp4"]
brightness_before = []
brightness_after = []

for video in video_files:
    print(f"\n🎬 Processing video: {video}")
    before, after = process_video(video, f"processed_{video}")
    brightness_before.append(before if before else 0)
    brightness_after.append(after if after else 0)

# ==== Generate Brightness Comparison Chart ====
x = np.arange(len(video_files))
width = 0.35

plt.figure(figsize=(10,6))
plt.bar(x - width/2, brightness_before, width, label='Before Enhancement', color='gray')
plt.bar(x + width/2, brightness_after, width, label='After Enhancement', color='orange')
plt.xticks(x, video_files)
plt.ylabel("Average Brightness")
plt.title("Video Brightness: Before vs After Enhancement")
plt.legend()
plt.tight_layout()
plt.savefig("Output/brightness_comparison_chart.png")
plt.show()
