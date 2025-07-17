import cv2
import subprocess
import json
from ultralytics import YOLO
from jetson_utils import cudaFromNumpy, videoOutput, cudaDrawRect, cudaFont

# Load YOLOv8 model
model = YOLO('runs/train/yolov8-shark_cam/weights/best.pt')  # Adjust path if needed

# Font for drawing text
font = cudaFont()

# Function to get direct stream URL from YouTube
def get_youtube_stream_url(youtube_url):
    cmd = ['yt-dlp', '-f', 'best[ext=mp4]', '--no-warnings', '-j', youtube_url]
    output = subprocess.check_output(cmd)
    metadata = json.loads(output)
    return metadata['url']

# Get YouTube livestream URL
youtube_page_url = 'https://www.youtube.com/watch?v=tEtg5Kg3voQ'
stream_url = get_youtube_stream_url(youtube_page_url)
print(f"🎥 Using livestream URL: {stream_url}")

# Open video stream
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("⚠️ Failed to open livestream.")
    exit(1)

# Set up WebRTC output
display = videoOutput("webrtc://@:8554/youtube")
print("ℹ️ WebRTC videoOutput initialized and waiting for connection...")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ Failed to grab frame, retrying...")
        continue

    # YOLOv8 inference
    results = model(frame)[0]

    # Convert to CUDA image
    cuda_img = cudaFromNumpy(frame)

    # Draw boxes and class names
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Draw bounding box
        cudaDrawRect(cuda_img, (x1, y1, x2, y2), (0, 255, 0, 128))
        # Draw class label
        font.OverlayText(cuda_img, text=class_name, x=x1 + 5, y=y1 + 5, color=(0, 255, 0, 128), background=(0, 0, 0, 160))

    print(f"✅ {len(results.boxes)} objects detected")

    # Display stream
    display.Render(cuda_img)
    if display.IsStreaming():
        display.SetStatus("✅ WebRTC client connected!")
    else:
        display.SetStatus("ℹ️ Waiting for WebRTC connection...")



