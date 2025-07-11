import cv2
import os

#Resize and overlay the video that she talks about her life as a YouTuber on the top left of each video.

input_folder = "Output/Task A/Section 2/"
talking_video = "Recorded Videos (4)/talking.mp4"
output_folder = "Output/Task A/Section 3/" 

overlay_size = (350,200) #Resize overlay video to 350x200 pixels
overlay_position = (30, 30) #Position of the overlay video on the top left corner

#Load the talking video
talking_vid = cv2.VideoCapture(talking_video)

#Detect fps of the talking video
talking_fps = talking_vid.get(cv2.CAP_PROP_FPS)

#Store each frame in a list after resizing
talking_frames = []

while True:
    ret, frame = talking_vid.read()
    if not ret:
        break
    resized_frame = cv2.resize(frame, overlay_size)
    talking_frames.append(resized_frame)

talking_vid.release() 
print(f"Loaded {len(talking_frames)} frames from the talking video.")

#Loop through each video in the input folder
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.mp4')):
        continue #Skip non-video files

    print(f"\nProcessing video: {filename}")
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, f"overlayed_{filename}")

    #Open the main video (already processed in section 1 and 2)
    vid = cv2.VideoCapture(input_path)

    #Get basic video properties
    main_fps = int(vid.get(cv2.CAP_PROP_FPS)) #frames per second
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)) #width of the video
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) #height of the video

    #Open video writer to save the output

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), main_fps, (width, height))

    frame_index = 0

    #Process each frame of the main video
    while True:
        ret, frame = vid.read() #Read a frame from the video
        if not ret:
            break 
                
        #Calculate the current time of frame in seconds
        current_time_sec = frame_index / main_fps

        #Calculate the corresponding frame index in the talking video
        talking_frame_index = int(current_time_sec * talking_fps)

        #Overlay the talking video frame if available
        if talking_frame_index < len(talking_frames):
            overlay_frame = talking_frames[talking_frame_index]
            ox, oy = overlay_position #overlay position
            oh, ow = overlay_frame.shape[:2] #overlay height and width

            #Simple overlay (no transparency)
            frame[oy:oy+oh, ox:ox+ow] = overlay_frame

        #Write the frame to the output video
        out.write(frame)
        frame_index += 1     
        
    #Release the video objects
    vid.release()
    out.release()
    print(f"Processed video saved as: {output_path}")     

print("\nAll videos processed with overlay successfully!")
