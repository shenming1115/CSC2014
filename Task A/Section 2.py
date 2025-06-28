import cv2
import numpy as np
import os

"""Blur all the faces (camera facing) that appear in a video."""

VIDEO_FOLDER = "Recorded Videos (4)/"
OUTPUT_FOLDER = "Output/Task A/Section 2/"
FACE_CASCADE_PATH = "Dependency/face_detector.xml"

# Automatically adapt to KCF or CSRT tracker
def create_tracker():
    if hasattr(cv2, 'legacy'):
        if hasattr(cv2.legacy, 'TrackerKCF_create'):
            return cv2.legacy.TrackerKCF_create()
        elif hasattr(cv2.legacy, 'TrackerCSRT_create'):
            return cv2.legacy.TrackerCSRT_create()
    else:
        if hasattr(cv2, 'TrackerKCF_create'):
            return cv2.TrackerKCF_create()
        elif hasattr(cv2, 'TrackerCSRT_create'):
            return cv2.TrackerCSRT_create()
    raise AttributeError("No supported tracker found! Please install opencv-contrib-python.")

def blur_faces_in_frame(frame, trackers):
    for tracker in trackers:
        ok, bbox = tracker.update(frame)
        if ok:
            x, y, w, h = [int(i) for i in bbox]
            # Edge handling to prevent out-of-bounds
            x, y = max(0, x), max(0, y)
            w, h = max(1, w), max(1, h)
            # Ratio-based dynamic blur kernel
            blur_size = (max(21, w//2*2+1), max(21, h//2*2+1))  # Always ensure odd size
            face_roi = frame[y:y+h, x:x+w]
            face_roi = cv2.GaussianBlur(face_roi, blur_size, 30)
            frame[y:y+h, x:x+w] = face_roi
    return frame

def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(40,40))
    return faces

def process_video(input_path, output_path, face_cascade):
    vid = cv2.VideoCapture(input_path)
    if not vid.isOpened():
        print(f"Failed to open video: {input_path}")
        return
     # Automatically select suitable codec and extension
    ext = os.path.splitext(output_path)[1].lower()
    if ext in ['.avi']:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_id = 0
    trackers = []
    detection_interval = 10 # Every 10 frames, re-detect faces and reset trackers

    while True:
        success, frame = vid.read()
        if not success:
            break

        # Every detection_interval frames, re-detect faces and reset trackers
        if frame_id % detection_interval == 0 or len(trackers) == 0:
            faces = detect_faces(frame, face_cascade)
            trackers = []
            for (x, y, w, h) in faces:
                tracker = create_tracker()
                tracker.init(frame, (x, y, w, h))
                trackers.append(tracker)

        frame = blur_faces_in_frame(frame, trackers)
        out.write(frame)
        frame_id += 1
        # Show progress every 100 frames
        if frame_id % 100 == 0:
            print(f"{input_path}: {frame_id} frames processed.")

    vid.release()
    out.release()
    print(f"Finished processing: {input_path}")

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    for filename in os.listdir(VIDEO_FOLDER):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            input_path = os.path.join(VIDEO_FOLDER, filename)
            output_path = os.path.join(OUTPUT_FOLDER, f"blurred_{filename}")
            process_video(input_path, output_path, face_cascade)

if __name__ == "__main__":
    main()
