from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import torch
import sys
import os

# Add src to path so we can import model modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model.predict import load_model, predict

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "model.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH, device)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Prediction endpoint will fail.")
    yield
    # Clean up resources if needed

app = FastAPI(title="Cats vs Dogs Qualifier", lifespan=lifespan)

@app.get("/health")
async def health_check():
    if model is None:
        return {"status": "unhealthy", "reason": "Model not loaded"}
    return {"status": "healthy"}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    try:
        contents = await file.read()
        label, confidence = predict(model, contents, device)
        return {
            "filename": file.filename,
            "prediction": label,
            "confidence": f"{confidence:.2f}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
