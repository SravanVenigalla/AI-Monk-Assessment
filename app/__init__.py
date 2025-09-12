from app.models import load_yolov3, load_yolov5, load_ssd

_models = {}

def get_model(model_name: str):
    """Return cached model instance or load it if not available."""
    model_name = model_name.lower()
    if model_name not in _models:
        if model_name == "yolov3":
            _models[model_name] = load_yolov3()
        elif model_name == "yolov5":
            _models[model_name] = load_yolov5()
        elif model_name == "ssd":
            _models[model_name] = load_ssd()
        else:
            raise ValueError(f"Model '{model_name}' not supported.")
    return _models[model_name]
