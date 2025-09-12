import os, uuid
from enum import Enum
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from app import get_model
from app.utils import preprocess_image, run_inference, save_annotated_image
from app.config import OUTPUT_DIR

class ModelName(str, Enum):
    yolov3 = "yolov3"
    yolov5 = "yolov5"
    ssd = "ssd"

app = FastAPI(title="Object Detection API")

@app.get("/")
def root():
    return {"message": "Object Detection API is running 🚀"}

@app.post("/detect/")
async def detect(file: UploadFile = File(...), model_name: ModelName = Form(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image = preprocess_image(contents)
    print(type(model_name.value))
    model = get_model(model_name.value)
    detections = run_inference(image, model_name.value, model)

    filename = f"{uuid.uuid4().hex}_{model_name.value}.jpg"
    save_annotated_image(image, detections, filename)

    return JSONResponse({
        "model": model_name.value,
        "image_filename": filename,
        "num_detections": len(detections),
        "detections": detections
    })

@app.get("/outputs/{filename}")
def get_output(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(file_path, media_type="image/jpeg")
