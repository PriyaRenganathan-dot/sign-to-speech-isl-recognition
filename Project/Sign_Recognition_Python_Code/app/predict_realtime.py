# ============================================================
#  STEP 3: REAL-TIME PREDICTION
#  Sign Recognition App — Final Year Project
#  Loads trained model → runs webcam → predicts sign → speaks
# ============================================================
#
#  HOW TO RUN:
#  pip install mediapipe opencv-python tensorflow pyttsx3 numpy
#  python predict_realtime.py
#
#  CONTROLS:
#  Q = Quit | S = Save to history | C = Clear output

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import time
import pyttsx3
import threading
import json
from collections import deque
from datetime import datetime

# ─── CONFIG ──────────────────────────────────────────────────
MODEL_PATH     = os.path.join('..', 'models', 'sign_model.h5')
LABELS_PATH    = os.path.join('..', 'models', 'labels.npy')
SEQUENCE_LEN   = 30
NUM_FEATURES   = 126
THRESHOLD      = 0.75       # Minimum confidence to show prediction
STABLE_FRAMES  = 16         # Frames needed to confirm sign
HISTORY_FILE   = 'history.json'

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

# ─── MEDIAPIPE ───────────────────────────────────────────────
mp_hands     = mp.solutions.hands
mp_drawing   = mp.solutions.drawing_utils
mp_draw_sty  = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.70,
)

# ─── TTS ENGINE (Shinchan-style) ─────────────────────────────
tts = pyttsx3.init()
tts.setProperty('rate', 230)    # Fast like Shinchan
tts.setProperty('volume', 1.0)
# Try to set high-pitched voice
voices = tts.getProperty('voices')
for v in voices:
    if 'female' in v.name.lower() or 'zira' in v.name.lower():
        tts.setProperty('voice', v.id)
        break

def speak_async(text):
    """Speak in background thread so camera doesn't freeze."""
    def _speak():
        tts.say(text)
        tts.runAndWait()
    t = threading.Thread(target=_speak, daemon=True)
    t.start()

# ─── EXTRACT KEYPOINTS ───────────────────────────────────────
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

# ─── DRAW LANDMARKS ──────────────────────────────────────────
def draw_landmarks(frame, results):
    if results.multi_hand_landmarks:
        for hl in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hl, mp_hands.HAND_CONNECTIONS,
                mp_draw_sty.get_default_hand_landmarks_style(),
                mp_draw_sty.get_default_hand_connections_style()
            )

# ─── DRAW HUD ────────────────────────────────────────────────
def draw_hud(frame, sign_en, sign_ta, conf, state, sentence, fps):
    H, W = frame.shape[:2]

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 90), (15, 23, 41), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    cv2.putText(frame, 'ISL SIGN RECOGNITION — LIVE',
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140, 200, 255), 1)
    cv2.putText(frame, f'FPS: {fps:.0f}',
                (W - 90, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 220, 160), 1)

    # Current prediction
    if sign_en:
        color = (100, 220, 160) if conf > 0.85 else (255, 200, 80)
        cv2.putText(frame, f'EN: {sign_en.upper()}',
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        # Confidence bar
        bar_w = int((W - 160) * conf)
        cv2.rectangle(frame, (10, 72), (W - 150, 80), (40, 50, 80), -1)
        cv2.rectangle(frame, (10, 72), (10 + bar_w, 80), color, -1)
        cv2.putText(frame, f'{conf*100:.0f}%',
                    (W - 140, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # State label (top-right)
    state_colors = {
        'detecting': (80, 180, 255),
        'confirmed': (100, 220, 160),
        'idle':      (120, 120, 150),
    }
    sc = state_colors.get(state, (150, 150, 150))
    cv2.putText(frame, state.upper(),
                (W - 140, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 2)

    # Bottom sentence panel
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, H - 70), (W, H), (15, 23, 41), -1)
    cv2.addWeighted(overlay2, 0.85, frame, 0.15, 0, frame)

    sent_display = ' '.join(sentence[-6:]) if sentence else '—'
    cv2.putText(frame, f'Sentence: {sent_display}',
                (10, H - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 220, 255), 1)

    # Controls hint
    cv2.putText(frame, 'Q:Quit  S:Save  C:Clear  SPACE:Speak',
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 120, 160), 1)

    return frame

# ─── SAVE HISTORY ────────────────────────────────────────────
def save_history(sign_en, sign_ta, conf):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
    history.append({
        'english': sign_en,
        'tamil':   sign_ta,
        'conf':    round(conf * 100, 1),
        'time':    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"  💾 Saved: {sign_en} ({sign_ta}) — {conf*100:.1f}%")

# ─── MAIN PREDICTION LOOP ────────────────────────────────────
def predict_realtime():
    # Load model
    print("Loading model...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        print("   Please run train_model.py first!")
        return

    model  = tf.keras.models.load_model(MODEL_PATH)
    labels = np.load(LABELS_PATH, allow_pickle=True)
    print(f"✅ Model loaded — {len(labels)} sign classes")
    print(f"   Classes: {list(labels)}")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    sequence   = deque(maxlen=SEQUENCE_LEN)  # Rolling window of frames
    sentence   = []                           # Confirmed words
    last_sign  = ''
    stable_cnt = 0
    state      = 'idle'
    cur_sign   = ''
    cur_conf   = 0.0
    fps_time   = time.time()
    fps        = 0
    frame_cnt  = 0

    print("\n" + "="*50)
    print("  REAL-TIME PREDICTION RUNNING")
    print("  Show your hand to the camera!")
    print("="*50)

    with hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            # Draw hand skeleton
            draw_landmarks(frame, results)

            # Extract keypoints
            kp = extract_keypoints(results)
            sequence.append(kp)

            # Only predict when we have a full sequence
            if len(sequence) == SEQUENCE_LEN:
                X = np.expand_dims(np.array(sequence), axis=0)  # (1, 30, 126)
                pred = model.predict(X, verbose=0)[0]

                top_idx  = np.argmax(pred)
                top_conf = float(pred[top_idx])
                top_sign = labels[top_idx]

                if top_conf >= THRESHOLD:
                    state    = 'detecting'
                    cur_sign = top_sign
                    cur_conf = top_conf

                    # Stable detection (STABLE_FRAMES consistent)
                    if top_sign == last_sign:
                        stable_cnt += 1
                    else:
                        last_sign  = top_sign
                        stable_cnt = 1

                    if stable_cnt >= STABLE_FRAMES:
                        state = 'confirmed'
                        ta = TAMIL_MAP.get(top_sign, top_sign)

                        # Add to sentence (avoid repeating same word)
                        if not sentence or sentence[-1] != top_sign:
                            sentence.append(top_sign)
                            print(f"  ✅ {top_sign.upper()} ({ta}) — {top_conf*100:.1f}%")
                            speak_async(top_sign)
                        stable_cnt = 0
                else:
                    state = 'idle'
                    if stable_cnt > 0:
                        stable_cnt = max(0, stable_cnt - 1)

            # FPS counter
            frame_cnt += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                fps      = frame_cnt / elapsed
                frame_cnt = 0
                fps_time  = time.time()

            # Draw HUD
            ta_cur = TAMIL_MAP.get(cur_sign, '') if state != 'idle' else ''
            frame  = draw_hud(frame, cur_sign if state != 'idle' else '',
                              ta_cur, cur_conf, state, sentence, fps)
            cv2.imshow('Sign Recognition — ISL', frame)

            # Key controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and cur_sign and state == 'confirmed':
                save_history(cur_sign, TAMIL_MAP.get(cur_sign, cur_sign), cur_conf)
            elif key == ord('c'):
                sentence.clear()
                cur_sign = ''
                cur_conf = 0
                state    = 'idle'
                print("  🗑 Output cleared")
            elif key == ord(' ') and sentence:
                speak_async(' '.join(sentence))

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Session ended")
    if sentence:
        print(f"  Final sentence: {' '.join(sentence)}")

if __name__ == '__main__':
    predict_realtime()
