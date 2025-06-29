import cv2
import numpy as np
import os

# Blur all the faces (camera facing) that appear in a video.

VIDEO_FOLDER = "Recorded Videos (4)/"
OUTPUT_FOLDER = "Output/Task A/Section 2/"
FACE_CASCADE_PATH = "Dependency/face_detector.xml"

# Auto-adapt KCF or CSRT tracker
def create_tracker():
    if hasattr(cv2, 'legacy'):
        # New OpenCV (>=4.5.2)
        if hasattr(cv2.legacy, 'TrackerKCF_create'):
            return cv2.legacy.TrackerKCF_create()
        elif hasattr(cv2.legacy, 'TrackerCSRT_create'):
            return cv2.legacy.TrackerCSRT_create()
    else:
        # Old OpenCV
        if hasattr(cv2, 'TrackerKCF_create'):
            return cv2.TrackerKCF_create()
        elif hasattr(cv2, 'TrackerCSRT_create'):
            return cv2.TrackerCSRT_create()
    raise AttributeError("No supported tracker found! Please install opencv-contrib-python.")

class FaceTracker:
    def __init__(self, bbox, frame, tolerance_frames=15):
        self.tracker = create_tracker()
        self.tracker.init(frame, bbox)
        self.last_bbox = bbox
        self.lost_frames = 0
        self.tolerance_frames = tolerance_frames  # Tolerance frames for tracking loss
        self.active = True
        self.confidence_history = []  # Tracking confidence history
        self.max_history = 10
        self.expand_ratio = 1.15  # Expansion ratio for blur region
        
        # Smooth tracking related
        self.bbox_history = [bbox]  # Position history
        self.velocity = (0, 0)  # Velocity vector (vx, vy)
        self.max_bbox_history = 5
        self.smooth_factor = 0.3  # Smoothing factor
        
    def update(self, frame):
        ok, bbox = self.tracker.update(frame)
        if ok:
            # Smooth process the new bounding box
            smoothed_bbox = self.smooth_bbox(bbox)
            
            # Update confidence history
            self.confidence_history.append(1.0)
            if len(self.confidence_history) > self.max_history:
                self.confidence_history.pop(0)
            
            # Update position history and velocity
            self.update_motion_model(smoothed_bbox)
            self.last_bbox = smoothed_bbox
            self.lost_frames = 0
            return True, smoothed_bbox
        else:
            # Record failed tracking
            self.confidence_history.append(0.0)
            if len(self.confidence_history) > self.max_history:
                self.confidence_history.pop(0)
                
            self.lost_frames += 1
            if self.lost_frames > self.tolerance_frames:
                self.active = False
                return False, None
            
            # During tolerance period, use motion predicted position
            predicted_bbox = self.predict_next_position()
            return True, predicted_bbox
    
    def smooth_bbox(self, new_bbox):
        # Smooth bounding box to reduce jitter
        if len(self.bbox_history) == 0:
            return new_bbox
        
        last_bbox = self.bbox_history[-1]
        x1, y1, w1, h1 = last_bbox
        x2, y2, w2, h2 = new_bbox
        
        # Use weighted average for smoothing
        smooth_x = x1 * (1 - self.smooth_factor) + x2 * self.smooth_factor
        smooth_y = y1 * (1 - self.smooth_factor) + y2 * self.smooth_factor
        smooth_w = w1 * (1 - self.smooth_factor) + w2 * self.smooth_factor
        smooth_h = h1 * (1 - self.smooth_factor) + h2 * self.smooth_factor
        
        return (smooth_x, smooth_y, smooth_w, smooth_h)
    
    def update_motion_model(self, bbox):
        # Update motion model, calculate velocity vector
        self.bbox_history.append(bbox)
        if len(self.bbox_history) > self.max_bbox_history:
            self.bbox_history.pop(0)
        
        # Calculate velocity vector (based on last two positions)
        if len(self.bbox_history) >= 2:
            curr_x, curr_y = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2
            prev_x, prev_y = (self.bbox_history[-2][0] + self.bbox_history[-2][2]/2, 
                             self.bbox_history[-2][1] + self.bbox_history[-2][3]/2)
            
            # Update velocity vector (with smoothing)
            new_vx = curr_x - prev_x
            new_vy = curr_y - prev_y
            self.velocity = (self.velocity[0] * 0.7 + new_vx * 0.3,
                           self.velocity[1] * 0.7 + new_vy * 0.3)
    
    def predict_next_position(self):
        # Predict next position based on motion model
        if not self.bbox_history:
            return self.get_expanded_bbox()
        
        last_bbox = self.bbox_history[-1]
        x, y, w, h = last_bbox
        
        # Use velocity vector to predict new position
        predicted_x = x + self.velocity[0] * self.lost_frames
        predicted_y = y + self.velocity[1] * self.lost_frames
        
        # Consider velocity decay (people don't move at constant speed)
        decay_factor = 0.95 ** self.lost_frames
        predicted_x = x + (predicted_x - x) * decay_factor
        predicted_y = y + (predicted_y - y) * decay_factor
        
        # Keep fixed expansion ratio, don't increase over time
        expand_ratio = 1.15  # Fixed expansion ratio
        expand_w = int(w * expand_ratio)
        expand_h = int(h * expand_ratio)
        expand_x = predicted_x - (expand_w - w) / 2
        expand_y = predicted_y - (expand_h - h) / 2
        
        return (expand_x, expand_y, expand_w, expand_h)

    def get_expanded_bbox(self):
        # Get expanded bounding box for handling profile faces or partial occlusion
        x, y, w, h = self.last_bbox
        # Expand bounding box
        expand_w = int(w * self.expand_ratio)
        expand_h = int(h * self.expand_ratio)
        expand_x = int(x - (expand_w - w) / 2)
        expand_y = int(y - (expand_h - h) / 2)
        return (expand_x, expand_y, expand_w, expand_h)
    
    def get_confidence(self):
        # Get average confidence of the tracker
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)

def blur_faces_in_frame(frame, face_trackers):
    for tracker in face_trackers[:]:  # Use copy for iteration to avoid modification errors
        ok, bbox = tracker.update(frame)
        if ok:
            x, y, w, h = [int(i) for i in bbox]
            # Boundary handling to prevent overflow
            x = max(0, min(x, frame.shape[1] - 1))
            y = max(0, min(y, frame.shape[0] - 1))
            w = max(1, min(w, frame.shape[1] - x))
            h = max(1, min(h, frame.shape[0] - y))
            
            # Adjust blur intensity based on tracking status
            blur_intensity = get_adaptive_blur_intensity(tracker)
            
            # Dynamically adjust blur kernel proportionally
            base_blur = max(21, w//3*2+1)
            blur_size = (base_blur, base_blur)  # Ensure odd number
            sigma = 30 * blur_intensity
            
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size > 0:
                face_roi = cv2.GaussianBlur(face_roi, blur_size, sigma)
                frame[y:y+h, x:x+w] = face_roi
        elif not tracker.active:
            # Remove inactive trackers
            face_trackers.remove(tracker)
    return frame

def get_adaptive_blur_intensity(tracker):
    # Adaptively adjust blur intensity based on tracker status
    confidence = tracker.get_confidence()
    lost_frames = tracker.lost_frames
    
    # Base intensity
    base_intensity = 1.0
    
    # Lower confidence leads to slightly stronger blur
    confidence_factor = 1.0 + (1.0 - confidence) * 0.2  # Reduced impact
    
    # Impact of lost frames: keep blur intensity relatively stable
    if lost_frames > 0:
        # Slightly weaken blur, but not too obvious
        fade_factor = max(0.8, 1.0 - lost_frames * 0.02)  # Gentler decay
    else:
        fade_factor = 1.0
    
    return base_intensity * confidence_factor * fade_factor

def validate_face_region(frame, bbox):
    # Validate whether the detected region is actually a face
    x, y, w, h = [int(v) for v in bbox]
    
    # Ensure bounding box is within image bounds
    x = max(0, min(x, frame.shape[1] - 1))
    y = max(0, min(y, frame.shape[0] - 1))
    w = max(1, min(w, frame.shape[1] - x))
    h = max(1, min(h, frame.shape[0] - y))
    
    # Extract candidate region
    face_region = frame[y:y+h, x:x+w]
    if face_region.size == 0:
        return False
    
    # Convert to grayscale
    if len(face_region.shape) == 3:
        gray_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    else:
        gray_region = face_region
    
    # 1. Check texture complexity of the region (faces usually have moderate texture)
    # Use Laplacian operator for edge detection
    laplacian = cv2.Laplacian(gray_region, cv2.CV_64F)
    texture_variance = laplacian.var()
    
    # Faces usually have moderate texture complexity (not too smooth or too complex)
    if texture_variance < 50 or texture_variance > 8000:
        return False
    
    # 2. Check brightness distribution (faces usually have relatively uniform brightness)
    mean_brightness = gray_region.mean()
    brightness_std = gray_region.std()
    
    # Exclude regions that are too dark, too bright, or have excessive brightness variation
    if mean_brightness < 20 or mean_brightness > 220 or brightness_std > 80:
        return False
    
    # 3. Check aspect ratio (already checked in detect_faces, reconfirm here)
    aspect_ratio = w / h
    if not (0.6 <= aspect_ratio <= 1.6):
        return False
    
    return True

def classify_scene_density(faces):
    # Classify scene density based on number of detected faces
    face_count = len(faces)
    if face_count <= 2:        # More conservative threshold
        return "low"  # Low density: 1-2 faces
    elif face_count <= 5:      # More conservative threshold
        return "medium"  # Medium density: 3-5 faces
    else:
        return "high"  # High density: 6+ faces

def merge_overlapping_faces(faces, threshold=0.4):
    # 合并重叠的人脸检测框，避免过度分割
    if len(faces) <= 1:
        return faces
    
    merged = []
    used = [False] * len(faces)
    
    for i, face1 in enumerate(faces):
        if used[i]:
            continue
            
        x1, y1, w1, h1 = face1
        merged_face = [x1, y1, x1+w1, y1+h1] # Convert to [x1, y1, x2, y2] format
        
        for j, face2 in enumerate(faces[i+1:], i+1):
            if used[j]:
                continue
                
            x2, y2, w2, h2 = face2
            # Calculate overlap
            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)
            
            if x_right > x_left and y_bottom > y_top:
                intersection = (x_right - x_left) * (y_bottom - y_top)
                area1 = w1 * h1
                area2 = w2 * h2
                union = area1 + area2 - intersection
                
                if intersection / union > threshold:
                    # Combine bounding boxes
                    merged_face[0] = min(merged_face[0], x2)
                    merged_face[1] = min(merged_face[1], y2)
                    merged_face[2] = max(merged_face[2], x2 + w2)
                    merged_face[3] = max(merged_face[3], y2 + h2)
                    used[j] = True
        
        # Convert back to [x, y, w, h] format
        merged.append((merged_face[0], merged_face[1], 
                      merged_face[2] - merged_face[0], 
                      merged_face[3] - merged_face[1]))
        used[i] = True
    
    return merged

def adaptive_blur_region(frame, bbox, density="low", tracker=None):
    # Based on scene density, adaptively adjust the blur region
    x, y, w, h = [int(v) for v in bbox]
    
    # Based on density, adjust the expansion ratio (fixed ratio, not time-dependent)
    if density == "high":
        expand_ratio = 1.25  # High density: expand blur region
    elif density == "medium":
        expand_ratio = 1.15
    else:
        expand_ratio = 1.1   # Low density: precise blur
    
    # Expand the bounding box
    expand_w = int(w * expand_ratio)
    expand_h = int(h * expand_ratio)
    expand_x = max(0, int(x - (expand_w - w) / 2))
    expand_y = max(0, int(y - (expand_h - h) / 2))
    
    # Ensure no out-of-bounds
    expand_x = max(0, min(expand_x, frame.shape[1] - 1))
    expand_y = max(0, min(expand_y, frame.shape[0] - 1))
    expand_w = max(1, min(expand_w, frame.shape[1] - expand_x))
    expand_h = max(1, min(expand_h, frame.shape[0] - expand_y))
    
    # Get adaptive blur intensity (keep relatively stable)
    if tracker:
        blur_intensity = get_adaptive_blur_intensity(tracker)
    else:
        blur_intensity = 1.0
    
    # Adaptive blur intensity
    if density == "high":
        blur_kernel = (max(31, expand_w//4*2+1), max(31, expand_h//4*2+1))
        sigma = max(20, 40 * blur_intensity)  # Ensure minimum blur strength
    else:
        blur_kernel = (max(21, expand_w//3*2+1), max(21, expand_h//3*2+1))
        sigma = max(15, 30 * blur_intensity)  # Ensure minimum blur strength
    
    face_roi = frame[expand_y:expand_y+expand_h, expand_x:expand_x+expand_w]
    if face_roi.size > 0:
        face_roi = cv2.GaussianBlur(face_roi, blur_kernel, sigma)
        frame[expand_y:expand_y+expand_h, expand_x:expand_x+expand_w] = face_roi
    
    return frame

def is_face_overlap(bbox1, bbox2, threshold=0.3):
    # Check if two face regions overlap
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return False
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area > threshold

def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use stricter parameters to reduce false positives
    # Main detector: use stricter parameters
    faces_main = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.15,      # More conservative scale factor
        minNeighbors=6,        # Increase minNeighbors to reduce false positives
        minSize=(40, 40),      # Minimum face size
        maxSize=(250, 250),    # Limit max size to avoid detecting buildings
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Supplementary detector: for possibly missed small faces
    faces_small = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        maxSize=(120, 120),    # Only detect small faces
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Merge detection results
    all_faces = list(faces_main) + list(faces_small)
    if len(all_faces) == 0:
        return []
    
    # Remove duplicate detections, use stricter threshold
    merged_faces = merge_overlapping_faces(all_faces, threshold=0.4)
    
    # Further filter: filter possible false positives based on aspect ratio and size
    filtered_faces = []
    for (x, y, w, h) in merged_faces:
        # Face aspect ratio is usually between 0.7-1.4
        aspect_ratio = w / h
        if 0.6 <= aspect_ratio <= 1.6:
            # Filter out overly large boxes (possibly buildings)
            face_area = w * h
            frame_area = gray.shape[0] * gray.shape[1]
            if face_area / frame_area < 0.25:  # Face should not occupy more than 25% of the frame
                filtered_faces.append((x, y, w, h))
    
    return filtered_faces

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
    face_trackers = []
    tolerance_frames = max(20, fps // 2)  # Increase tolerance time, at least 20 frames
    scene_density = "low"  # Scene density
    high_density_mode = False  # High density mode flag
    last_detection_faces_count = 0
    
    while True:
        success, frame = vid.read()
        if not success:
            break

        # Adaptive detection interval
        if high_density_mode:
            detection_interval = 12  # More frequent detection in high density scenes
        else:
            detection_interval = 18  # Less frequent detection in normal scenes

        # Periodically re-detect faces, or when there are no active trackers
        active_trackers = [t for t in face_trackers if t.active]
        should_detect = (frame_id % detection_interval == 0 or 
                        len(active_trackers) == 0 or
                        len(active_trackers) < last_detection_faces_count * 0.7)  # If tracker count drops significantly
        
        if should_detect:
            faces = detect_faces(frame, face_cascade)
            
            # Validate detected faces, filter out false positives
            validated_faces = []
            for face in faces:
                if validate_face_region(frame, face):
                    validated_faces.append(face)
            
            faces = validated_faces
            last_detection_faces_count = len(faces)
            scene_density = classify_scene_density(faces)
            
            # Enable high density mode (lower threshold)
            if scene_density == "high":
                high_density_mode = True
                tolerance_frames = max(25, fps // 1.8)  # Increase tolerance in high density
            else:
                high_density_mode = False
                tolerance_frames = max(20, fps // 2)
            
            # For each detected face, check if a tracker is already tracking it
            for (x, y, w, h) in faces:
                new_bbox = (x, y, w, h)
                is_new_face = True
                overlap_threshold = 0.35 if scene_density == "high" else 0.25  # Stricter overlap threshold
                
                # Check if overlaps with existing trackers
                for tracker in face_trackers:
                    if tracker.active and is_face_overlap(new_bbox, tracker.last_bbox, overlap_threshold):
                        is_new_face = False
                        # If tracker confidence is low, reinitialize
                        if tracker.get_confidence() < 0.7:  # Raise confidence threshold
                            tracker.tracker = create_tracker()
                            tracker.tracker.init(frame, new_bbox)
                            # Keep motion history continuity
                            tracker.update_motion_model(new_bbox)
                            tracker.lost_frames = 0
                        break
                
                # If it's a new face, create a new tracker
                if is_new_face:
                    new_tracker = FaceTracker(new_bbox, frame, tolerance_frames)
                    face_trackers.append(new_tracker)

        # Choose blur strategy based on scene density
        if scene_density == "high":
            # High density scene: use adaptive blur
            for tracker in face_trackers[:]:
                ok, bbox = tracker.update(frame)
                if ok:
                    frame = adaptive_blur_region(frame, bbox, scene_density, tracker)
                elif not tracker.active:
                    face_trackers.remove(tracker)
        else:
            # Low/medium density: use original blur method
            frame = blur_faces_in_frame(frame, face_trackers)
            
        out.write(frame)
        frame_id += 1
        
        # Progress prompt
        if frame_id % 100 == 0:
            active_count = len([t for t in face_trackers if t.active])
            print(f"{input_path}: {frame_id} frames processed, {active_count} active trackers, density: {scene_density}")

    vid.release()
    out.release()
    print(f"Finished processing: {input_path}")
    print(f"Total frames processed: {frame_id}")

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
