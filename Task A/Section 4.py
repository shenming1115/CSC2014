import cv2
import os

print("OpenCV version:", cv2.__version__)
print("Current working directory:", os.getcwd())

# === Paths ===
input_folder = "Output/Task A/Section 3/"
ending_video = "Recorded Videos (4)/Watermark & Ending/endscreen.mp4"
img1_path = "Recorded Videos (4)/Watermark & Ending/watermark1.png"
img2_path = "Recorded Videos (4)/Watermark & Ending/watermark2.png"
output_folder = "Output/Task A/Final_Output_Videos/"

# === Load watermarks ===
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
if img1 is None or img2 is None:
    print("Failed to load one of the watermark images.")
    exit()
else:
    print(f"Watermark1 loaded: {img1.shape}")
    print(f"Watermark2 loaded: {img2.shape}")

def watermarking(frame, counter):
    try:
        if counter < 100:
            watermark = img1
        else:
            watermark = img2
        alpha = 0.3  
        result = cv2.addWeighted(frame, 1-alpha, watermark, alpha, 0)
        return result
        
    except Exception as e:
        print(f"Error in watermarking: {e}")
        return frame

def process_video(input_path, output_path):
    global img1, img2
    vid = cv2.VideoCapture(input_path)
    if not vid.isOpened():
        print(f"Failed to open video: {input_path}")
        return

    fps = int(vid.get(cv2.CAP_PROP_FPS))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Processing: {input_path}")
    print(f"Original video size: {width}x{height}")
    
    # Ensure watermark size matches video
    img1_resized = cv2.resize(img1, (1280, 720))
    img2_resized = cv2.resize(img2, (1280, 720))
    
    print(f"Watermarks resized to: {img1_resized.shape}")

    
    img1, img2 = img1_resized, img2_resized

    # Output setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1280, 720))

    counter = 0
    while True:
        ret, frame = vid.read()
        if not ret or frame is None:
            break

        if width != 1280 or height != 720:
            frame = cv2.resize(frame, (1280, 720))

        frame = watermarking(frame, counter)
        out.write(frame)
        counter += 1

    vid.release()

    # === Add ending video ===
    end = cv2.VideoCapture(ending_video)
    if end.isOpened():
        while True:
            ret, end_frame = end.read()
            if not ret:
                break
            end_frame = cv2.resize(end_frame, (1280, 720))
            out.write(end_frame)
        end.release()

    out.release()
    print(f"Done: {output_path}")

# === Main Execution ===
def main():
    print("Starting main function...")
    
    if not os.path.exists(input_folder):
        print(f"Input folder not found: {input_folder}")
        return

    if not os.path.exists(output_folder):
        print(f"Creating output folder: {output_folder}")
        os.makedirs(output_folder)

    videos = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not videos:
        print("No video files found.")
        return

    print(f"Found {len(videos)} video files: {videos}")
    
    for v in videos:
        input_path = os.path.join(input_folder, v)
        output_path = os.path.join(output_folder, f"watermarked_{v}")
        print(f"Processing: {v}")
        process_video(input_path, output_path)
    
    print("All videos processed!")

if __name__ == "__main__":
    print("Script started...")
    main()
    print("Script completed.")
