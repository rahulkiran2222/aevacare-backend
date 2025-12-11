from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime

# --- CONFIGURATION ---
app = FastAPI()

# UPDATE THIS LIST WITH YOUR CLOUDFLARE DOMAINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "https://aevacare.pages.dev",
        "https://aevacare.horizontrax.com" 
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI ENGINE SETUP ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

class AnalysisResponse(BaseModel):
    category: str
    confidence: float
    explanation: str
    ui_color: str

def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def analyze_face_logic(image):
    """
    REAL COMPUTER VISION LOGIC
    Analyzes Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR)
    """
    height, width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return "No Face Detected", 0.0, "Camera clear? No face found.", "yellow"

    landmarks = results.multi_face_landmarks[0].landmark

    # 1. Calculate Eye Openness (EAR)
    # Indices for Left Eye (Top/Bottom) and Right Eye
    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[145]
    right_eye_top = landmarks[386]
    right_eye_bottom = landmarks[374]
    
    avg_eye_openness = (calculate_distance(left_eye_top, left_eye_bottom) + 
                        calculate_distance(right_eye_top, right_eye_bottom)) / 2

    # 2. Calculate Mouth Openness (MAR)
    mouth_top = landmarks[13]
    mouth_bottom = landmarks[14]
    mouth_openness = calculate_distance(mouth_top, mouth_bottom)

    # 3. Brow Furrow (Distance between brows) - approximated
    brow_left = landmarks[107]
    brow_right = landmarks[336]
    brow_dist = calculate_distance(brow_left, brow_right)

    # --- RULES FOR PAIN/DISTRESS ---
    # These thresholds are heuristic-based
    
    # CONDITION 1: Crying / Screaming (Mouth wide open + Eyes squeezed or open)
    if mouth_openness > 0.08: 
        if avg_eye_openness < 0.015:
            return "High Pain / Distress", 0.92, "Eyes squeezed shut and mouth wide open (Crying pattern).", "red"
        else:
            return "Agitated / Vocalizing", 0.85, "Mouth is open wide (possible crying or shouting).", "red"

    # CONDITION 2: Grimace (Eyes squeezed shut tight, mouth closed)
    if avg_eye_openness < 0.012:
        return "Pain / Discomfort", 0.75, "Eyes are tightly squeezed shut (Grimace detected).", "yellow"

    # CONDITION 3: Neutral / Calm
    if avg_eye_openness > 0.02 and mouth_openness < 0.05:
        return "Calm / Comfortable", 0.88, "Eyes open, mouth relaxed. Neutral expression.", "green"

    # Fallback
    return "Mild Discomfort / Uncertain", 0.50, "Subtle signs detected. Please check patient manually.", "yellow"


@app.post("/analyze/frame", response_model=AnalysisResponse)
async def analyze_frame(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        # Resize for speed (max 640px wide)
        h, w = img.shape[:2]
        if w > 640:
            scale = 640 / w
            img = cv2.resize(img, (640, int(h * scale)))

        category, conf, explain, color = analyze_face_logic(img)

        return {
            "category": category,
            "confidence": conf,
            "explanation": explain,
            "ui_color": color
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "category": "Error",
            "confidence": 0.0,
            "explanation": "Could not process image.",
            "ui_color": "yellow"
        }

@app.get("/")
def health_check():
    return {"status": "AevaCare AI Engine Online"}
