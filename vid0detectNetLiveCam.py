import cv2
import subprocess
import json
from ultralytics import YOLO
from jetson_utils import cudaFromNumpy, videoOutput, cudaDrawRect
import torch

model = torch.load('your_model.pt', map_location='cpu', weights_only=False)

# Get the direct stream URL using yt-dlp
def get_youtube_stream_url(youtube_url):
    cmd = ['yt-dlp', '-f', 'best[ext=mp4]', '--no-warnings', '-j', youtube_url]
    output = subprocess.check_output(cmd)
    metadata = json.loads(output)
    return metadata['url']

# Load the YOLOv8 model (GPU will be used automatically if available)
model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt depending on Jetson capability

# Retrieve livestream URL
youtube_page_url = 'https://www.youtube.com/watch?v=tEtg5Kg3voQ'
stream_url = get_youtube_stream_url(youtube_page_url)
print(f"🎥 Using livestream URL: {stream_url}")

# Open the livestream
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

    # Run YOLOv8 inference
    results = model(frame)[0]

    # Convert frame to CUDA image
    cuda_img = cudaFromNumpy(frame)

    # Draw each bounding box
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cudaDrawRect(cuda_img, (x1, y1, x2, y2), (255, 0, 0, 255))

    print(f"✅ {len(results.boxes)} objects detected")

    # Stream to WebRTC
    display.Render(cuda_img)
    if display.IsStreaming():
        display.SetStatus("✅ WebRTC client connected!")
    else:
        display.SetStatus("ℹ️ Waiting for WebRTC connection...")




