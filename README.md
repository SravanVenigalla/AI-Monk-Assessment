# 🖼️ Object Detection Microservice

This project is a **microservice-based object detection API** built with **FastAPI**.
It supports multiple lightweight detection models (**YOLOv3, YOLOv5, SSD**) and returns results in a structured JSON format.
Images are uploaded via an API endpoint, and bounding boxes are drawn + returned along with metadata.

---

## 📌 Features

* Upload an image and run object detection with your choice of model (`yolov3`, `yolov5`, `ssd`).
* Returns:

  * **JSON output** → detected classes, confidence scores, bounding boxes.
  * **Processed image** → with bounding boxes drawn (optional).
* Modular design:

  * `models.py` → model loading functions
  * `utils.py` → inference + postprocessing
  * `main.py` → FastAPI entrypoint

---

## 📂 Project Structure

```
AI-Monk-Assessment/
│── app/
│   ├── __init__.py      # model cache & manager
│   ├── models.py        # model loaders (YOLOv3, YOLOv5, SSD)
│   ├── utils.py         # inference logic & post-processing
│   ├── config.py        # device (CPU/GPU) config
├── main.py              # FastAPI app entrypoint
│── requirements.txt
│── Dockerfile
│── README.md
```

---

## ⚙️ Prerequisites

* Python 3.9+
* [Docker](https://docs.docker.com/get-docker/) (optional, for containerized runs)
* (Optional) GPU with CUDA for acceleration; otherwise runs on CPU.

---

## 🚀 Setup & Usage

### 🔹 1. Clone Repo

```bash
git clone https://github.com/SravanVenigalla/AI-Monk-Assessment.git
cd AI-Monk-Assessment
```

### 🔹 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔹 3. Run Locally

```bash
uvicorn main:app --reload
```

Visit Swagger UI at 👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🐳 Run with Docker

### 🔹 Build

```bash
docker build -t object-detection-app .
```

### 🔹 Run

```bash
docker run -p 8000:8000 object-detection-app
```

Then open 👉 [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📡 API Endpoint

### **POST** `/detect/`

#### Parameters:

* `file` → image file (UploadFile)
* `model_name` → string (`yolov3`, `yolov5`, `ssd`)

#### Example (cURL)

```bash
curl -X POST "http://localhost:8000/detect/" \
  -F "file=@sample.jpg" \
  -F "model_name=yolov5"
```

#### Example Response (JSON)

```json
{
  "model": "yolov5",
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.89,
      "bbox": [34, 57, 120, 150]
    },
    {
      "class_id": 2,
      "class_name": "car",
      "confidence": 0.76,
      "bbox": [200, 80, 340, 200]
    }
  ]
}
```

## 📚 References

* [Ultralytics YOLOv3](https://github.com/ultralytics/yolov3)
* [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
* [TorchVision SSD](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)
* [FastAPI](https://fastapi.tiangolo.com/)

---
