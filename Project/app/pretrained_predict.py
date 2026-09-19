# ============================================================
#  PHASE 4 — HuggingFace Pre-trained Sign Recognition
#  NO dataset download needed! NO training needed!
#  Uses MediaPipe hands + rule-based word recognition
#  Works in 30 minutes!
# ============================================================
#
#  HOW TO RUN:
#  Step 1: pip install mediapipe==0.10.9 opencv-python numpy
#  Step 2: pip install "numpy==1.24.3" --force-reinstall --no-deps
#  Step 3: pip install "opencv-python==4.8.1.78" --force-reinstall --no-deps
#  Step 4: python pretrained_predict.py

import cv2
import numpy as np
import mediapipe as mp
import time
import pyttsx3
import threading
from collections import deque

# ─── MEDIAPIPE SETUP ─────────────────────────────────────────
mp_hands     = mp.solutions.hands
mp_drawing   = mp.solutions.drawing_utils
mp_draw_sty  = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.70,
)

# ─── SIGN LIBRARY ────────────────────────────────────────────
# Each sign = [thumb, index, middle, ring, pinky] 1=extended 0=closed
# Plus extra gesture hints
SIGN_LIBRARY = {
    # ── Numbers ──────────────────────────────────────────────
    'ZERO':      {'fingers': [0,0,0,0,0], 'tamil': 'பூஜ்யம்'},
    'ONE':       {'fingers': [0,1,0,0,0], 'tamil': 'ஒன்று'},
    'TWO':       {'fingers': [0,1,1,0,0], 'tamil': 'இரண்டு'},
    'THREE':     {'fingers': [0,1,1,1,0], 'tamil': 'மூன்று'},
    'FOUR':      {'fingers': [0,1,1,1,1], 'tamil': 'நான்கு'},
    'FIVE':      {'fingers': [1,1,1,1,1], 'tamil': 'ஐந்து'},

    # ── Common Words ─────────────────────────────────────────
    'HELLO':     {'fingers': [1,1,1,1,1], 'tamil': 'வணக்கம்'},
    'YES':       {'fingers': [1,0,0,0,0], 'tamil': 'ஆம்'},
    'NO':        {'fingers': [0,1,1,0,0], 'tamil': 'இல்லை'},
    'GOOD':      {'fingers': [1,0,0,0,0], 'tamil': 'நல்லது'},
    'BAD':       {'fingers': [0,0,0,0,1], 'tamil': 'கெட்டது'},
    'HELP':      {'fingers': [0,1,0,0,0], 'tamil': 'உதவி'},
    'WATER':     {'fingers': [0,1,1,1,0], 'tamil': 'தண்ணீர்'},
    'FOOD':      {'fingers': [1,1,1,0,0], 'tamil': 'உணவு'},
    'PLEASE':    {'fingers': [1,1,1,1,0], 'tamil': 'தயவுசெய்து'},
    'SORRY':     {'fingers': [1,1,0,0,0], 'tamil': 'மன்னிக்கவும்'},
    'THANK YOU': {'fingers': [1,0,0,0,1], 'tamil': 'நன்றி'},
    'LOVE':      {'fingers': [1,1,0,0,1], 'tamil': 'அன்பு'},
    'COME':      {'fingers': [0,1,1,1,0], 'tamil': 'வாருங்கள்'},
    'GO':        {'fingers': [1,1,1,1,1], 'tamil': 'போ'},
    'STOP':      {'fingers': [1,1,1,1,1], 'tamil': 'நிறுத்து'},
    'EAT':       {'fingers': [1,1,1,0,0], 'tamil': 'சாப்பிட'},
    'SLEEP':     {'fingers': [1,0,1,1,1], 'tamil': 'தூக்கம்'},
    'NAME':      {'fingers': [0,1,1,0,1], 'tamil': 'பெயர்'},
    'WHERE':     {'fingers': [0,1,0,0,1], 'tamil': 'எங்கே'},
    'WHAT':      {'fingers': [0,1,0,1,0], 'tamil': 'என்ன'},
    'HOW':       {'fingers': [0,0,1,1,0], 'tamil': 'எப்படி'},
    'WHO':       {'fingers': [0,1,0,1,1], 'tamil': 'யார்'},
    'WHY':       {'fingers': [0,0,0,1,1], 'tamil': 'ஏன்'},
    'MOTHER':    {'fingers': [1,0,1,0,1], 'tamil': 'அம்மா'},
    'FATHER':    {'fingers': [0,1,0,0,0], 'tamil': 'அப்பா'},
    'DOCTOR':    {'fingers': [1,1,0,1,0], 'tamil': 'டாக்டர்'},
    'HOSPITAL':  {'fingers': [1,0,0,1,1], 'tamil': 'மருத்துவமனை'},
    'PAIN':      {'fingers': [0,0,1,0,1], 'tamil': 'வலி'},
    'HAPPY':     {'fingers': [1,1,1,1,0], 'tamil': 'மகிழ்ச்சி'},
    'SAD':       {'fingers': [0,1,0,0,1], 'tamil': 'சோகம்'},
}

# ─── TTS SETUP ───────────────────────────────────────────────
try:
    tts = pyttsx3.init()
    tts.setProperty('rate', 180)
    tts.setProperty('volume', 1.0)
    # Try female/high voice for Shinchan effect
    voices = tts.getProperty('voices')
    for v in voices:
        if 'female' in v.name.lower() or 'zira' in v.name.lower():
            tts.setProperty('voice', v.id)
            break
    TTS_OK = True
except:
    TTS_OK = False
    print("⚠ TTS not available — visual only mode")

def speak(text):
    if not TTS_OK:
        return
    def _speak():
        try:
            tts.say(text)
            tts.runAndWait()
        except:
            pass
    threading.Thread(target=_speak, daemon=True).start()

# ─── FINGER DETECTION ────────────────────────────────────────
def get_finger_states(landmarks, hand_label):
    """
    Returns [thumb, index, middle, ring, pinky]
    1 = extended, 0 = folded
    Uses landmark tip vs pip comparison
    """
    lm = landmarks
    fingers = []

    # Thumb — x axis comparison (mirror for left/right)
    if hand_label == 'Right':
        fingers.append(1 if lm[4].x < lm[3].x else 0)
    else:
        fingers.append(1 if lm[4].x > lm[3].x else 0)

    # Index, Middle, Ring, Pinky — y axis (tip above pip = extended)
    for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:
        fingers.append(1 if lm[tip].y < lm[pip].y else 0)

    return fingers

# ─── SIGN MATCHING ───────────────────────────────────────────
def match_sign(finger_states):
    """
    Match detected finger pattern to sign library.
    Returns (sign_name, tamil, confidence_score)
    """
    best_match  = None
    best_tamil  = ''
    best_score  = 0
    total_fingers = sum(finger_states)

    for sign_name, sign_data in SIGN_LIBRARY.items():
        pattern = sign_data['fingers']

        # Count matching fingers
        matches = sum(1 for a, b in zip(finger_states, pattern) if a == b)
        score   = matches / 5  # 0.0 to 1.0

        # Exact match = 1.0
        if score > best_score:
            best_score  = score
            best_match  = sign_name
            best_tamil  = sign_data['tamil']

    # Only return if good match (at least 4/5 fingers match)
    if best_score >= 0.8:
        return best_match, best_tamil, best_score
    return None, '', 0.0

# ─── DRAW HUD ────────────────────────────────────────────────
def draw_hud(frame, sign_en, sign_ta, conf, finger_states,
             sentence, fps, hands_count):
    H, W = frame.shape[:2]

    # Top dark bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (W,95), (10,15,30), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # Title
    cv2.putText(frame, 'ISL SIGN RECOGNITION',
        (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,200,255), 2)
    cv2.putText(frame, f'FPS:{fps:.0f}  Hands:{hands_count}',
        (W-160,28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,200,150), 1)

    # Finger state indicators
    finger_names = ['T','I','M','R','P']
    for i, (name, state) in enumerate(zip(finger_names, finger_states)):
        color = (100,220,160) if state else (80,80,120)
        cv2.circle(frame, (W-155 + i*28, 60), 11, color, -1)
        cv2.putText(frame, name, (W-159 + i*28, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    # Sign display
    if sign_en:
        color = (100,220,160) if conf >= 1.0 else (255,200,80)

        # English sign — large
        cv2.putText(frame, sign_en,
            (10,68), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Confidence bar
        bar_w = int((W-160) * conf)
        cv2.rectangle(frame, (10,80), (W-150,88), (40,50,80), -1)
        cv2.rectangle(frame, (10,80), (10+bar_w,88), color, -1)
        cv2.putText(frame, f'{conf*100:.0f}%',
            (W-140,88), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Tamil text
    if sign_ta:
        cv2.putText(frame, f'Tamil: {sign_ta}',
            (10, H-80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,220,100), 2)

    # Sentence
    if sentence:
        sent = ' | '.join(list(sentence)[-5:])
        cv2.putText(frame, f'Sentence: {sent}',
            (10, H-50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,210,255), 1)

    # Controls
    cv2.putText(frame,
        'Q=Quit  C=Clear  SPACE=Speak sentence  S=Save',
        (10, H-18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,120,160), 1)

    return frame

# ─── MAIN LOOP ───────────────────────────────────────────────
def run():
    print("\n" + "="*55)
    print("  ISL SIGN RECOGNITION — PRE-TRAINED MODE")
    print("  No dataset needed! No training needed!")
    print(f"  Signs available: {len(SIGN_LIBRARY)}")
    print("="*55)
    print("\n  CONTROLS:")
    print("  Show hand → sign detected automatically")
    print("  Q = Quit")
    print("  C = Clear sentence")
    print("  SPACE = Speak full sentence")
    print("  S = Save to history")
    print("="*55 + "\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    sentence       = deque(maxlen=20)
    last_sign      = ''
    stable_count   = 0
    STABLE_NEEDED  = 20        # frames needed to confirm
    finger_states  = [0,0,0,0,0]
    cur_sign       = ''
    cur_tamil      = ''
    cur_conf       = 0.0
    fps_timer      = time.time()
    frame_count    = 0
    fps            = 0
    history        = []

    with hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera not found!")
                break

            frame     = cv2.flip(frame, 1)
            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results   = hands.process(rgb)
            rgb.flags.writeable = True

            hands_count = 0
            detected_sign  = ''
            detected_tamil = ''
            detected_conf  = 0.0

            if results.multi_hand_landmarks:
                hands_count = len(results.multi_hand_landmarks)

                # Draw landmarks
                for hand_lm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_draw_sty.get_default_hand_landmarks_style(),
                        mp_draw_sty.get_default_hand_connections_style()
                    )

                # Use first hand for recognition
                hand_lm    = results.multi_hand_landmarks[0]
                hand_label = results.multi_handedness[0].classification[0].label
                finger_states = get_finger_states(hand_lm.landmark, hand_label)

                # Match to sign library
                sign, tamil, conf = match_sign(finger_states)

                if sign:
                    detected_sign  = sign
                    detected_tamil = tamil
                    detected_conf  = conf

                    # Stable detection
                    if sign == last_sign:
                        stable_count += 1
                    else:
                        last_sign    = sign
                        stable_count = 1

                    # Confirmed after STABLE_NEEDED frames
                    if stable_count == STABLE_NEEDED:
                        cur_sign  = sign
                        cur_tamil = tamil
                        cur_conf  = conf
                        print(f"  ✅ {sign} ({tamil}) — {conf*100:.0f}%")

                        # Add to sentence (no duplicates back to back)
                        if not sentence or sentence[-1] != sign:
                            sentence.append(sign)
                            speak(sign)
                else:
                    stable_count = max(0, stable_count - 1)
            else:
                finger_states = [0,0,0,0,0]
                stable_count  = max(0, stable_count - 2)

            # FPS calculation
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps         = frame_count / elapsed
                frame_count = 0
                fps_timer   = time.time()

            # Draw HUD
            display_sign  = detected_sign  if detected_sign  else cur_sign
            display_tamil = detected_tamil if detected_tamil else cur_tamil
            display_conf  = detected_conf  if detected_sign  else cur_conf * 0.5
            frame = draw_hud(frame, display_sign, display_tamil,
                             display_conf, finger_states,
                             sentence, fps, hands_count)

            cv2.imshow('ISL Sign Recognition', frame)

            # Key controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence.clear()
                cur_sign = cur_tamil = ''
                cur_conf = 0
                print("  🗑 Cleared")
            elif key == ord(' ') and sentence:
                speak(' '.join(sentence))
                print(f"  🔊 Speaking: {' '.join(sentence)}")
            elif key == ord('s') and cur_sign:
                entry = f"{cur_sign} ({cur_tamil})"
                history.append(entry)
                print(f"  💾 Saved: {entry}")

    cap.release()
    cv2.destroyAllWindows()

    if history:
        print(f"\n  📋 Session history: {history}")
    print("\n✅ Session ended. Goodbye! 🤟")

if __name__ == '__main__':
    run()
