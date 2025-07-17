import os
import shutil
import random
import yaml
from ultralytics import YOLO

# ⚙️ Configuration
ROOT = "YOLOv8_dataset"  # base dataset folder
IMG_EXT = ".jpg"         # image file extension
VAL_SPLIT = 0.2          # 20% of data for validation
EPOCHS = 50
IMG_SIZE = 640
BATCH = 4

def make_splits():
    # Read class names
    with open(os.path.join(ROOT, "classes.txt")) as f:
        names = [x.strip() for x in f if x.strip()]
    nc = len(names)

    # List all image files
    imgs = [f for f in os.listdir(os.path.join(ROOT, "images")) if f.endswith(IMG_EXT)]
    random.shuffle(imgs)
    n_val = int(len(imgs) * VAL_SPLIT)
    val_imgs, train_imgs = imgs[:n_val], imgs[n_val:]

    # Prepare train/val directories
    for split in ["train", "val"]:
        for sub in ["images", "labels"]:
            os.makedirs(os.path.join(ROOT, split, sub), exist_ok=True)

    # Copy images and labels to train/val folders
    for split, file_list in [("train", train_imgs), ("val", val_imgs)]:
        for img in file_list:
            stem = os.path.splitext(img)[0]
            shutil.copy(os.path.join(ROOT, "images", img),
                        os.path.join(ROOT, split, "images", img))
            label_src = os.path.join(ROOT, "labels", stem + ".txt")
            label_dst = os.path.join(ROOT, split, "labels", stem + ".txt")
            if os.path.exists(label_src):
                shutil.copy(label_src, label_dst)

    # Create data.yaml with absolute paths
    data_yaml = {
        "train": os.path.abspath(os.path.join(ROOT, "train", "images")),
        "val": os.path.abspath(os.path.join(ROOT, "val", "images")),
        "nc": nc,
        "names": names
    }
    with open(os.path.join(ROOT, "data.yaml"), "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print("🔧 Created data.yaml with classes:", names)
    print("📁 Training images:", len(train_imgs), "| Validation images:", len(val_imgs))


def train():
    model = YOLO("yolov8n.pt")  # load a pretrained nano model
    model.train(
        data=os.path.abspath(os.path.join(ROOT, "data.yaml")),
        imgsz=IMG_SIZE,
        batch=BATCH,
        epochs=EPOCHS,
        device=0,       # set to "cpu" if you're not using a GPU
        half=True,      # use FP16 if supported
        workers=2,
        project="runs/train",
        name="yolov8-shark_cam",
        exist_ok=True
    )
    model.val(data=os.path.abspath(os.path.join(ROOT, "data.yaml")), imgsz=IMG_SIZE)
    print("✅ Training complete. Best model at runs/train/yolov8-shark_cam/weights/best.pt")


if __name__ == "__main__":
    random.seed(42)
    make_splits()
    train()
