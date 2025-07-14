import cv2
import subprocess
import json
from jetson_inference import detectNet
from jetson_utils import cudaFromNumpy, videoOutput, cudaDrawRect


def get_youtube_stream_url(youtube_url):
    cmd = ['yt-dlp', '-f', 'best[ext=mp4]', '--no-warnings', '-j', youtube_url]
    output = subprocess.check_output(cmd)
    metadata = json.loads(output)
    return metadata['url']


youtube_page_url = 'https://www.youtube.com/watch?v=w3LjpFhySTg'
stream_url = get_youtube_stream_url(youtube_page_url)
print(f"🎥 Using livestream URL: {stream_url}")

cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("⚠️ Failed to open livestream.")
    exit(1)

net = detectNet("ssd-mobilenet-v2", threshold=0.5)

display = videoOutput("webrtc://@:8554/youtube")
print("ℹ️ WebRTC videoOutput initialized and waiting for connection...")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ Failed to grab frame, retrying...")
        continue  # Instead of break, retry reading frames

    cuda_img = cudaFromNumpy(frame)
    detections = net.Detect(cuda_img)

    for det in detections:
        x1, y1, x2, y2 = int(det.Left), int(det.Top), int(det.Right), int(det.Bottom)
        cudaDrawRect(cuda_img, (x1, y1, x2, y2), (255, 0, 0, 255))

    print(f"✅ {len(detections)} objects detected")

    display.Render(cuda_img)

    if display.IsStreaming():
        display.SetStatus("✅ WebRTC client connected!")
    else:
        display.SetStatus("ℹ️ Waiting for WebRTC connection...")

# Note: Add cleanup handling (e.g., try/except KeyboardInterrupt) if needed.




