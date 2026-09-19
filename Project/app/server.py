# ============================================================
#  STEP 4: FastAPI BACKEND SERVER
#  Sign Recognition App — Final Year Project
#  Connects trained Python model to the HTML frontend
# ============================================================
#
#  HOW TO RUN:
#  pip install fastapi uvicorn tensorflow mediapipe opencv-python numpy pillow python-multipart
#  python server.py
#  → Opens at http://localhost:8000
#  → API docs at http://localhost:8000/docs

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import numpy as np
import cv2
import base64
import json
import os
import time
import mediapipe as mp
import tensorflow as tf
from collections import deque
from typing import Optional
import io
from PIL import Image
import uvicorn
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────
MODEL_PATH    = os.path.join('..', 'models', 'sign_model.h5')
LABELS_PATH   = os.path.join('..', 'models', 'labels.npy')
SEQUENCE_LEN  = 30
NUM_FEATURES  = 126
THRESHOLD     = 0.72
STABLE_FRAMES = 16

# ─── TAMIL MAP ───────────────────────────────────────────────
TAMIL_MAP = {
    'hello':'வணக்கம்',     'thank_you':'நன்றி',
    'yes':'ஆம்',            'no':'இல்லை',
    'sorry':'மன்னிக்கவும்', 'please':'தயவுசெய்து',
    'love':'அன்பு',         'help':'உதவி',
    'water':'தண்ணீர்',      'good':'நல்லது',
    'bad':'கெட்டது',        'eat':'சாப்பிட',
    'sleep':'தூக்கம்',      'come':'வாருங்கள்',
    'go':'போ',              'zero':'பூஜ்யம்',
    'one':'ஒன்று',          'two':'இரண்டு',
    'three':'மூன்று',       'four':'நான்கு',
    'five':'ஐந்து',         'name':'பெயர்',
    'where':'எங்கே',        'how':'எப்படி',
    'what':'என்ன',
}

# ─── FastAPI APP ─────────────────────────────────────────────
app = FastAPI(
    title="Sign Recognition API",
    description="ISL Sign Language Recognition — Final Year Project",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── GLOBAL STATE ────────────────────────────────────────────
class AppState:
    model      = None
    labels     = None
    mp_hands   = None
    hands      = None
    mp_drawing = None
    sequence   = deque(maxlen=SEQUENCE_LEN)
    last_sign  = ''
    stable_cnt = 0
    sessions   = {}   # websocket → session data

state = AppState()

# ─── STARTUP ─────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    print("🚀 Starting Sign Recognition API...")

    # Load MediaPipe
    state.mp_hands   = mp.solutions.hands
    state.mp_drawing = mp.solutions.drawing_utils
    state.hands = state.mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.70,
        static_image_mode=False,
    )

    # Load model (if exists)
    if os.path.exists(MODEL_PATH):
        print("  Loading trained model...")
        state.model  = tf.keras.models.load_model(MODEL_PATH)
        state.labels = np.load(LABELS_PATH, allow_pickle=True)
        print(f"  ✅ Model loaded — {len(state.labels)} classes")
    else:
        print("  ⚠ Model not found — using finger counting fallback")

    print("✅ API Ready at http://localhost:8000")

# ─── KEYPOINT EXTRACTION ─────────────────────────────────────
def extract_keypoints(results):
    lh = np.zeros(63)
    rh = np.zeros(63)
    if results.multi_hand_landmarks:
        for i, lm in enumerate(results.multi_hand_landmarks):
            if i >= 2:
                break
            label = results.multi_handedness[i].classification[0].label
            kp = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
            if label == 'Left':
                lh = kp
            else:
                rh = kp
    return np.concatenate([lh, rh])

# ─── FINGER COUNT FALLBACK ───────────────────────────────────
def count_fingers(landmarks, hand_label):
    tips = [4, 8, 12, 16, 20]
    count = 0
    if hand_label == 'Right':
        if landmarks[4].x < landmarks[3].x:
            count += 1
    else:
        if landmarks[4].x > landmarks[3].x:
            count += 1
    for tip, pip in [(8,6),(12,10),(16,14),(20,18)]:
        if landmarks[tip].y < landmarks[pip].y:
            count += 1
    return min(count, 5)

NUM_WORDS = {0:'zero',1:'one',2:'two',3:'three',4:'four',5:'five'}

def fallback_predict(results):
    """Finger-count fallback when ML model not loaded."""
    if not results.multi_hand_landmarks:
        return None, 0.0, 0, 0

    label     = results.multi_handedness[0].classification[0].label
    score     = results.multi_handedness[0].classification[0].score
    count     = count_fingers(results.multi_hand_landmarks[0].landmark, label)
    word      = NUM_WORDS[count]
    tamil     = TAMIL_MAP[word]
    hands_cnt = len(results.multi_hand_landmarks)
    return word, float(score), count, hands_cnt

# ─── PROCESS FRAME ───────────────────────────────────────────
def process_frame(frame_bgr):
    """Run MediaPipe + model prediction on single BGR frame."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = state.hands.process(rgb)

    # Hand landmarks list
    landmarks_data = []
    hands_count    = 0
    hand_labels    = []

    if results.multi_hand_landmarks:
        hands_count = len(results.multi_hand_landmarks)
        for i, lm in enumerate(results.multi_hand_landmarks):
            lbl = results.multi_handedness[i].classification[0].label
            sc  = results.multi_handedness[i].classification[0].score
            pts = [[p.x, p.y, p.z] for p in lm.landmark]
            landmarks_data.append({'label': lbl, 'score': float(sc), 'points': pts})
            hand_labels.append(lbl)

    # Extract keypoints + add to rolling sequence
    kp = extract_keypoints(results)
    state.sequence.append(kp)

    prediction = None
    confidence = 0.0
    finger_count = 0

    if state.model and len(state.sequence) == SEQUENCE_LEN:
        # ── ML MODEL PREDICTION ──
        X    = np.expand_dims(np.array(state.sequence), axis=0)
        pred = state.model.predict(X, verbose=0)[0]
        idx  = int(np.argmax(pred))
        conf = float(pred[idx])

        if conf >= THRESHOLD:
            sign = str(state.labels[idx])
            if sign == state.last_sign:
                state.stable_cnt += 1
            else:
                state.last_sign   = sign
                state.stable_cnt  = 1
            if state.stable_cnt >= STABLE_FRAMES:
                prediction   = sign
                confidence   = conf
                state.stable_cnt = 0

        # Finger count from first hand
        if results.multi_hand_landmarks:
            lbl = results.multi_handedness[0].classification[0].label
            finger_count = count_fingers(results.multi_hand_landmarks[0].landmark, lbl)

    elif results.multi_hand_landmarks:
        # ── FALLBACK: finger counting ──
        lbl = results.multi_handedness[0].classification[0].label
        finger_count = count_fingers(results.multi_hand_landmarks[0].landmark, lbl)
        confidence   = float(results.multi_handedness[0].classification[0].score)

        if finger_count == state.last_sign if isinstance(state.last_sign, int) else False:
            state.stable_cnt += 1
        else:
            state.last_sign  = finger_count
            state.stable_cnt = 1

        if state.stable_cnt >= STABLE_FRAMES:
            prediction       = NUM_WORDS[finger_count]
            state.stable_cnt = 0

    tamil = TAMIL_MAP.get(prediction, '') if prediction else ''

    return {
        'detected':     prediction is not None,
        'sign_english': prediction or '',
        'sign_tamil':   tamil,
        'confidence':   round(confidence * 100, 1),
        'finger_count': finger_count,
        'hands_count':  hands_count,
        'hand_labels':  hand_labels,
        'landmarks':    landmarks_data,
        'model_active': state.model is not None,
    }

# ─── REST ENDPOINTS ──────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html><body style="font-family:sans-serif;padding:2rem;background:#f0f4ff">
    <h1>🤟 Sign Recognition API</h1>
    <p>ISL Sign Language Recognition — Final Year Project</p>
    <ul>
      <li><a href="/docs">📖 API Documentation</a></li>
      <li><a href="/status">📊 Server Status</a></li>
      <li><a href="/signs">📋 Supported Signs</a></li>
    </ul>
    <p><strong>WebSocket:</strong> ws://localhost:8000/ws/predict</p>
    </body></html>
    """

@app.get("/status")
async def status():
    return {
        "status":       "running",
        "model_loaded": state.model is not None,
        "labels_count": len(state.labels) if state.labels is not None else 0,
        "labels":       list(state.labels) if state.labels is not None else [],
        "mode":         "ML Model" if state.model else "Finger Count Fallback",
        "sequence_len": len(state.sequence),
    }

@app.get("/signs")
async def get_signs():
    return {
        "total": len(TAMIL_MAP),
        "signs": [
            {"english": k, "tamil": v}
            for k, v in TAMIL_MAP.items()
        ]
    }

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Predict sign from uploaded image."""
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    result = process_frame(frame)
    return JSONResponse(result)

# ─── WEBSOCKET — Real-time stream ────────────────────────────
@app.websocket("/ws/predict")
async def websocket_predict(ws: WebSocket):
    await ws.accept()
    client = ws.client
    print(f"  📡 WebSocket connected: {client}")
    state.sequence.clear()
    state.last_sign  = ''
    state.stable_cnt = 0

    try:
        while True:
            data = await ws.receive_text()
            msg  = json.loads(data)

            if msg.get('type') == 'frame':
                # Decode base64 image from browser
                img_data = base64.b64decode(msg['frame'].split(',')[-1])
                nparr    = np.frombuffer(img_data, np.uint8)
                frame    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None:
                    result = process_frame(frame)
                    await ws.send_text(json.dumps(result))

            elif msg.get('type') == 'reset':
                state.sequence.clear()
                state.last_sign  = ''
                state.stable_cnt = 0
                await ws.send_text(json.dumps({'type': 'reset_ok'}))

    except WebSocketDisconnect:
        print(f"  📡 WebSocket disconnected: {client}")
    except Exception as e:
        print(f"  ❌ WebSocket error: {e}")

# ─── RUN ─────────────────────────────────────────────────────
if __name__ == '__main__':
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
