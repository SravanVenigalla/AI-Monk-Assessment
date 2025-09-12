import torch
import torchvision
from app.config import DEVICE

def load_yolov3():
    print(f"Loading YOLOv3 on {DEVICE} ...")
    model = torch.hub.load("ultralytics/yolov3", "yolov3", pretrained=True)
    model.eval()
    print("YOLOv3 loaded successfully.")
    return model

def load_yolov5():
    print(f"Loading YOLOv5 on {DEVICE} ...")
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True).to(DEVICE)
    model.eval()
    print("YOLOv5 loaded successfully.")
    return model

def load_ssd():
    print(f"Loading SSD (TorchVision) on {DEVICE} ...")
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True).to(DEVICE)
    model.eval()
    print("SSD loaded successfully.")
    return model
