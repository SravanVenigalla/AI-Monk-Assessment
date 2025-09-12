import io, os
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from app.config import OUTPUT_DIR, DEVICE, COCO_CLASSES
import torch, uuid

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

def preprocess_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def run_inference(image, model_name: str, model):
    detections = []

    if model_name == "ssd":
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(tensor)[0]
        for i, box in enumerate(outputs["boxes"]):
            score = outputs["scores"][i].item()
            if score < 0.5:
                continue
            x1, y1, x2, y2 = box.cpu().numpy()
            cls_id = outputs["labels"][i].item()
            detections.append({
                "class_id": cls_id,
                "class_name": COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id),
                "confidence": float(score),
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })

    elif model_name in ["yolov3", "yolov5"]:
        results = model(image)
        for *box, conf, cls in results.xyxy[0].tolist():
            detections.append({
                "class_id": int(cls),
                "class_name": results.names[int(cls)],
                "confidence": float(conf),
                "bbox": [float(x) for x in box]
            })

    return detections

def save_annotated_image(image: Image.Image, detections, filename: str) -> str:
    out_path = os.path.join(OUTPUT_DIR, filename)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class_name']} {det['confidence']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, max(0, y1 - 10)), label, fill="red", font=font)
    image.save(out_path)
    return out_path