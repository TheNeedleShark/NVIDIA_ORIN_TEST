import os
import shutil
import random
import yaml
from ultralytics import YOLO

# ⚙️ Configuration
ROOT = "YOLOv8_dataset"     # base dataset folder
IMG_EXT = ".jpg"            # image file extension
VAL_SPLIT = 0.2             # 20% validation
EPOCHS = 50
IMG_SIZE = 640
BATCH = 4

def make_splits():
    with open(os.path.join(ROOT, "classes.txt")) as f:
        names = [x.strip() for x in f if x.strip()]
    nc = len(names)

    imgs = [f for f in os.listdir(os.path.join(ROOT, "images")) if f.endswith(IMG_EXT)]
    random.shuffle(imgs)
    n_val = int(len(imgs) * VAL_SPLIT)
    val_imgs, train_imgs = imgs[:n_val], imgs[n_val:]

    for split in ["train", "val"]:
        for sub in ["images", "labels"]:
            os.makedirs(os.path.join(ROOT, split, sub), exist_ok=True)

    for split, file_list in [("train", train_imgs), ("val", val_imgs)]:
        for img in file_list:
            stem = os.path.splitext(img)[0]
            shutil.copy(os.path.join(ROOT, "images", img),
                        os.path.join(ROOT, split, "images", img))
            label_src = os.path.join(ROOT, "labels", stem + ".txt")
            label_dst = os.path.join(ROOT, split, "labels", stem + ".txt")
            if os.path.exists(label_src):
                shutil.copy(label_src, label_dst)

    data_yaml = {
        "train": os.path.abspath(os.path.join(ROOT, "train", "images")),
        "val": os.path.abspath(os.path.join(ROOT, "val", "images")),
        "nc": nc,
        "names": names
    }
    with open(os.path.join(ROOT, "data.yaml"), "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print("🔧 Created data.yaml — classes:", names)
    print("📁 Training:", len(train_imgs), "| Validation:", len(val_imgs))

def cleanup_cache():
    # Remove any old dataset cache folders
    for root, dirs, files in os.walk(ROOT):
        for d in dirs:
            if d.endswith(".cache"):
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
    # Optional global Ultralytics cache
    u_cache = os.path.expanduser("~/.cache/ultralytics")
    if os.path.exists(u_cache):
        shutil.rmtree(u_cache, ignore_errors=True)
    print("🧹 Cleaned old cache directories")

def train():
    cleanup_cache()
    model = YOLO("yolov8n.pt")
    model.train(
        data=os.path.abspath(os.path.join(ROOT, "data.yaml")),
        imgsz=IMG_SIZE,
        batch=BATCH,
        epochs=EPOCHS,
        device=0,
        half=True,
        workers=2,
        project="runs/train",
        name="yolov8-shark_cam",
        exist_ok=True,
        cache=False  # disable dataset caching :contentReference[oaicite:1]{index=1}
    )
    model.val(
        data=os.path.abspath(os.path.join(ROOT, "data.yaml")),
        imgsz=IMG_SIZE,
        cache=False
    )
    print("✅ Training complete. Best model at runs/train/yolov8-shark_cam/weights/best.pt")

if __name__ == "__main__":
    random.seed(42)
    make_splits()
    train()
