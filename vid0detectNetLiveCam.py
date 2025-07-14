import cv2
from jetson_inference import detectNet
from jetson_utils import cudaFromNumpy, videoOutput, cudaDrawRect, cudaToNumpy

# Your livestream .m3u8 URL
youtube_stream_url = (
    "https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1752533608/ei/CDZ1aNeCH-yQsfIPjqyLsQY/..."
)

# GStreamer pipeline from YouTube
gst_str = (
    f'souphttpsrc location="{youtube_stream_url}" ! '
    'hlsdemux ! tsdemux ! h264parse ! nvv4l2decoder ! '
    'nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! '
    'video/x-raw, format=BGR ! appsink'
)

# Open GStreamer pipeline as video input
cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("⚠️ Failed to open livestream.")
    exit(1)

# Load DetectNet model
net = detectNet("ssd-mobilenet-v2", threshold=0.5)

# WebRTC stream output
display = videoOutput("webrtc://@:8554/livestream")  # can be accessed via browser

# Main detection loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame read failed. Retrying...")
        continue

    cuda_img = cudaFromNumpy(frame)
    detections = net.Detect(cuda_img)

    # Draw bounding boxes
    for det in detections:
        x1, y1, x2, y2 = int(det.Left), int(det.Top), int(det.Right), int(det.Bottom)
        cudaDrawRect(cuda_img, (x1, y1, x2, y2), (255, 0, 0, 255))  # red box

    print(f"✅ {len(detections)} objects detected")

    # Stream to WebRTC
    if display.IsStreaming():
        display.Render(cuda_img)
        display.SetStatus("DetectNet | {:.0f} FPS".format(net.GetNetworkFPS()))
    else:
        print("ℹ️ Waiting for WebRTC connection...")



