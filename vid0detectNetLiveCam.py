import torch
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential, ModuleList
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv

torch.serialization.add_safe_globals([DetectionModel])
torch.serialization.add_safe_globals([Sequential])
torch.serialization.add_safe_globals([DetectionModel, Conv, Sequential, ModuleList])

import cv2
import subprocess
import json
from ultralytics import YOLO
from jetson_utils import cudaFromNumpy, videoOutput, cudaDrawRect

print(torch.version.cuda)      # should show CUDA version, not None
print(torch.cuda.is_available())  # should return True

# 1. Get direct stream URL via yt-dlp
def get_youtube_stream_url(youtube_url):
    cmd = ['yt-dlp', '-f', 'best[ext=mp4]', '--no-warnings', '-j', youtube_url]
    output = subprocess.check_output(cmd)
    metadata = json.loads(output)
    return metadata['url']

# 2. Export YOLOv8 model to TensorRT engine (run ONCE)
onnx_engine = 'yolov8n.engine'
model = YOLO('yolov8n.pt')
print("🚀 Exporting model to TensorRT engine (this may take a moment)...")
model.export(format='engine')  # creates yolov8n.engine :contentReference[oaicite:1]{index=1}

# 3. Load the TensorRT-optimized model
trt_model = YOLO(onnx_engine)

# 4. Get livestream URL
youtube_page_url = 'https://www.youtube.com/watch?v=tEtg5Kg3voQ'
stream_url = get_youtube_stream_url(youtube_page_url)
print(f"🎥 Using livestream URL: {stream_url}")

# 5. Open stream via OpenCV
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print("⚠️ Failed to open livestream.")
    exit(1)

# 6. Initialize WebRTC output
display = videoOutput("webrtc://@:8554/youtube")
print("ℹ️ WebRTC videoOutput initialized, awaiting connection...")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ Failed to grab frame, retrying...")
        continue

    # 7. GPU-based inference using TensorRT engine
    results = trt_model(frame)

    # 8. Draw detection boxes via CUDA
    cuda_img = cudaFromNumpy(frame)
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cudaDrawRect(cuda_img, (x1, y1, x2, y2), (0, 255, 0, 255))

    print(f"✅ {len(results.boxes)} objects detected")

    # 9. Stream output via WebRTC
    display.Render(cuda_img)
    if display.IsStreaming():
        display.SetStatus("✅ WebRTC client connected!")
    else:
        display.SetStatus("ℹ️ Waiting for WebRTC connection...")




