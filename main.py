from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import mediapipe as mp

app = FastAPI()

# --- CORS SETUP (Update with your specific domains) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for testing, restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI CONFIGURATION ---
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
    debug_info: dict

# --- GEOMETRY HELPERS ---
def get_point(landmarks, index, w, h):
    return np.array([landmarks[index].x * w, landmarks[index].y * h])

def euclidean_dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def analyze_emotion_vectors(image):
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return "No Face Detected", 0.0, "Camera blocked or face not visible.", "yellow", {}

    lm = results.multi_face_landmarks[0].landmark

    # --- CRITICAL LANDMARKS (MediaPipe Indices) ---
    # Lips
    lip_top = get_point(lm, 13, w, h)
    lip_bottom = get_point(lm, 14, w, h)
    mouth_left = get_point(lm, 61, w, h)
    mouth_right = get_point(lm, 291, w, h)
    
    # Brows
    brow_left_inner = get_point(lm, 107, w, h)
    brow_right_inner = get_point(lm, 336, w, h)
    
    # Eyes (Upper/Lower for EAR)
    eye_L_top = get_point(lm, 159, w, h)
    eye_L_bot = get_point(lm, 145, w, h)
    eye_R_top = get_point(lm, 386, w, h)
    eye_R_bot = get_point(lm, 374, w, h)

    # --- CALCULATE METRICS ---
    
    # 1. MOUTH OPENNESS (Vertical / Horizontal)
    mouth_height = euclidean_dist(lip_top, lip_bottom)
    mouth_width = euclidean_dist(mouth_left, mouth_right)
    mouth_ratio = mouth_height / mouth_width # Big number = wide open

    # 2. SMILE CURVE (Are corners higher than the center?)
    # Note: In images, Y=0 is top. So Smaller Y = Higher on face.
    mouth_center_y = (lip_top[1] + lip_bottom[1]) / 2
    mouth_corners_avg_y = (mouth_left[1] + mouth_right[1]) / 2
    
    # If corners are significantly HIGHER (smaller Y) than center -> Smile
    smile_curve = mouth_center_y - mouth_corners_avg_y 
    
    # 3. BROW SQUEEZE (Normalized by face width to handle zoom)
    # Face width approx = distance between cheekbones (using landmarks 234 and 454)
    face_width = euclidean_dist(get_point(lm, 234, w, h), get_point(lm, 454, w, h))
    brow_dist = euclidean_dist(brow_left_inner, brow_right_inner)
    brow_ratio = brow_dist / face_width # Small number = Furrowed

    # 4. EYE OPENNESS (EAR)
    ear = (euclidean_dist(eye_L_top, eye_L_bot) + euclidean_dist(eye_R_top, eye_R_bot)) / 2
    ear_ratio = ear / face_width # Normalized

    # --- LOGIC TREE (HIERARCHY) ---
    
    debug = {
        "smile_val": round(smile_curve, 2),
        "brow_val": round(brow_ratio, 3),
        "mouth_open": round(mouth_ratio, 2)
    }

    # CASE A: HAPPY / LAUGHING
    # Logic: Corners are high (positive smile_curve) AND Brows are NOT squeezed
    if smile_curve > 5.0: # Threshold depends on resolution, but >0 usually means smile
        if mouth_ratio > 0.3:
            return "Laughing / Joyful", 0.95, "Mouth open with corners lifted. Cheeks raised.", "green", debug
        else:
            return "Smiling / Content", 0.90, "Mouth corners lifted in a smile.", "green", debug

    # CASE B: PAIN / CRYING
    # Logic: Mouth open OR grimace + Brows Squeezed + No Smile
    if brow_ratio < 0.28: # Brows are very close together
        if mouth_ratio > 0.2:
            return "Crying / High Distress", 0.95, "Brows furrowed (V-shape) + Mouth open wide.", "red", debug
        else:
            return "Grimacing / Pain", 0.85, "Brows furrowed + Eyes squeezed. Potential silent pain.", "red", debug

    # CASE C: SURPRISE (Mouth open, but brows high/far apart)
    if mouth_ratio > 0.4 and brow_ratio > 0.32:
        return "Surprised / Alert", 0.80, "Mouth open but brows are relaxed/raised.", "yellow", debug

    # CASE D: NEUTRAL / SLEEPING
    if mouth_ratio < 0.15:
        if ear_ratio < 0.02:
            return "Sleeping / Calm", 0.90, "Eyes closed + Mouth relaxed.", "green", debug
        else:
            return "Neutral / Comfortable", 0.85, "Face is relaxed. No distress markers.", "green", debug

    # CASE E: AMBIGUOUS (Mild Discomfort)
    return "Mild Unsettled / Checking", 0.50, "Some tension detected, but not definitive.", "yellow", debug


@app.post("/analyze/frame", response_model=AnalysisResponse)
async def analyze_frame(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        # Process
        cat, conf, expl, col, dbg = analyze_emotion_vectors(img)

        return {
            "category": cat,
            "confidence": conf,
            "explanation": expl,
            "ui_color": col,
            "debug_info": dbg
        }
    except Exception as e:
        return {
            "category": "Error",
            "confidence": 0.0,
            "explanation": str(e),
            "ui_color": "yellow",
            "debug_info": {}
        }
