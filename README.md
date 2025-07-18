# 🦈 YOLOv8 Monterrey Bay Aquarium Shark Cam Detection on Jetson Orin Nano

link to the stream: 
```
https://www.youtube.com/watch?v=tEtg5Kg3voQ
```

[![Jetson Nano](https://img.shields.io/badge/Jetson-Orin%20Nano-green?logo=nvidia)](https://developer.nvidia.com/embedded/jetson-orin)
[![JetPack](https://img.shields.io/badge/JetPack-6.2-blue?logo=nvidia)](https://developer.nvidia.com/embedded/jetpack)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-orange)](https://github.com/ultralytics/ultralytics)


This project uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) to detect various fish and shark species from YouTube livestreams in real time. It's optimized for the NVIDIA Jetson Orin Nano running **JetPack 6.2** and includes support for WebRTC video output.

## 📦 Features

- ✅ Real-time inference from YouTube livestreams
- ✅ Trained on a custom shark/fish species dataset
- ✅ WebRTC live stream output via Jetson's `jetson-utils`
- ✅ Automated dataset split and `data.yaml` generation
- ✅ On-device YOLOv8 training with GPU acceleration

---

## 🚀 Requirements

### Mandatory

- **NVIDIA Jetson Orin Nano**
- **JetPack SDK 6.2** (includes CUDA, cuDNN, TensorRT, VPI, and DeepStream)
- Python 3.10+
- pip with virtualenv or venv recommended

### Python Dependencies

Install with:

```bash
sudo apt update
sudo apt install libjpeg-dev libtiff5-dev libpng-dev python3-pip ffmpeg
pip3 install --upgrade pip
pip3 install ultralytics opencv-python-headless pyyaml yt-dlp
```


### install jetson utils and jetson-inference

# Clone and build jetson-inference (which includes jetson-utils)
```
git clone --recursive https://github.com/dusty-nv/jetson-inference
cd jetson-inference
mkdir build && cd build
cmake ../
make -j$(nproc)
sudo make install
sudo ldconfig
```





!!!!!!!!! DATASET STRUCTURE !!!!!!!!!!!!
### make sure it looks like this: 
```
YOLOv8_dataset/
├── classes.txt          # List of class names (one per line)
├── images/              # Raw image files
├── labels/              # YOLO-format .txt label files
```


### to train the model, run :
```
python3 training_yolov8.py
```

### make sure the dataset folder is in the same directory as this file


### when running the main file, make sure you see these two
```
✅ 3 objects detected
ℹ️ WebRTC videoOutput initialized and waiting for connection...
```
### then you are ready to connect!
### the stream should be at
```
http://<ip_address>:8554/youtube
```
### if it doesnt work remove '/youtube'

# Now you should be all set!











