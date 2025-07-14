import cv2
import subprocess
import json
from jetson_inference import detectNet
from jetson_utils import cudaFromNumpy, videoOutput, cudaDrawRect

# Use yt-dlp to extract livestream URL
def get_youtube_stream_url(youtube_url):
    cmd = ['yt-dlp', '-f', 'best[ext=mp4]', '--no-warnings', '-j', youtube_url]
    output = subprocess.check_output(cmd)
    metadata = json.loads(output)
    return metadata['url']

# Replace this with the actual YouTube livestream URL
youtube_page_url = 'https://www.youtube.com/watch?v=w3LjpFhySTg'
stream_url = get_youtube_stream_url(youtube_page_url)

# Open stream
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("⚠️ Failed to open livestream.")
    exit(1)

# Load model and WebRTC output
net = detectNet("ssd-mobilenet-v2", threshold=0.5)
display = videoOutput("webrtc://@:8554/youtube")

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame read failed.")
        continue

    cuda_img = cudaFromNumpy(frame)
    detections = net.Detect(cuda_img)

    for det in detections:
        x1, y1, x2, y2 = int(det.Left), int(det.Top), int(det.Right), int(det.Bottom)
        cudaDrawRect(cuda_img, (x1, y1, x2, y2), (255, 0, 0, 255))

    print(f"✅ {len(detections)} objects detected")

    if display.IsStreaming():
        display.Render(cuda_img)
        display.SetStatus("DetectNet | {:.0f} FPS".format(net.GetNetworkFPS()))
    else:
        print("ℹ️ Waiting for WebRTC connection...")
