# FastAPI
from fastapi import FastAPI, HTTPException, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import ast

# Preprocessing Layer
from app.preprocessing_layer import preprocess_image

# Model Loading
import torch

# Request Processing
import io

# .env loading
import os
from dotenv import load_dotenv


# Instantiate and prepare the app
app = FastAPI()

# load environment variables
load_dotenv()
model_path = os.getenv("MODEL_PATH")
classes = ast.literal_eval(os.getenv("CLASSES"))

# Prepare static and templates paths
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Set the device and verify model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check model existence
def check_model_existance(model_path):
    if not os.path.exists(model_path):
        model_name = os.path.basename(model_path)
        raise HTTPException(status_code=404, detail=f"Predictor {model_name} is not found at {model_path}")

# Load the predictor
check_model_existance(model_path=model_path)
predictor = torch.jit.load(model_path).to(device)
predictor.eval()

# Load index
@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        "index.html", context={"request":request}
    )

# Prediction logic
@app.post("/predict")
async def pred(file: UploadFile):
    image_content = await file.read()
    image_stream = io.BytesIO(image_content)
    image = preprocess_image(image_stream).to(device)
        
    with torch.inference_mode():
        pred_idx = torch.softmax(predictor(image), dim=1).argmax(dim=1).item()
        pred_class = classes[pred_idx]
    return {"response": pred_class}

@app.get("/health-check")
def health_check():
    return HTMLResponse(status_code=204)

@app.get("/ready")
def readiness_check():
    return HTMLResponse(status_code=204)