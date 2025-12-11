from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import random

app = FastAPI(
    title="AevaCare API",
    description="Experimental Facial Expression & Behaviour Insight Assistant",
    version="0.1.0"
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisResponse(BaseModel):
    timestamp: str
    status: str
    category: str
    confidence: float
    explanation: str
    disclaimer: str
    ui_color: str # 'green', 'yellow', 'red'

def mock_neonate_inference(image_array):
    """
    MOCK MODEL: In a real system, this would be a PyTorch/TF model 
    analyzing Brow Bulge, Eye Squeeze, Nasolabial Furrow.
    """
    # Simulating a prediction for demonstration
    pain_probability = random.uniform(0, 1) 
    
    if pain_probability < 0.4:
        return "Comfortable / Calm", "green", "Face appears relaxed. No stress markers detected.", pain_probability
    elif pain_probability < 0.7:
        return "Mild Discomfort", "yellow", "Slight brow furrowing detected. Monitor closely.", pain_probability
    else:
        return "Likely Pain / Distress", "red", "High intensity indicators: Eye squeeze and open mouth detected.", pain_probability

@app.get("/")
def health_check():
    return {"status": "AevaCare system active", "mode": "experimental"}

@app.post("/analyze/frame", response_model=AnalysisResponse)
async def analyze_frame(file: UploadFile = File(...)):
    # 1. Read Image
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    # 2. Perform Inference (Mocked for Prototype)
    # In production: Detect face -> Crop -> Feed to CNN
    category, color, explanation, score = mock_neonate_inference(img)
    
    # 3. Construct Safety & Output
    from datetime import datetime
    return {
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "category": category,
        "confidence": round(score, 2),
        "explanation": explanation,
        "ui_color": color,
        "disclaimer": "RESEARCH PROTOTYPE ONLY. NOT A DIAGNOSIS. IF CONCERNED, CONTACT A CLINICIAN."
    }

# Instructions to run:
# pip install fastapi uvicorn opencv-python-headless multipart
# uvicorn main:app --reload
