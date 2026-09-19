# ============================================================
#  STEP 1: DATA COLLECTION
#  Sign Recognition App — Final Year Project
#  Collects hand landmark data for each sign using webcam
# ============================================================
#
#  HOW TO RUN:
#  pip install mediapipe opencv-python numpy
#  python collect_data.py
#
#  WHAT IT DOES:
#  - Opens webcam
#  - For each sign, records 30 sequences × 30 frames
#  - Saves 21 hand landmarks (x,y,z) per frame as numpy arrays
#  - Output: dataset/sign_name/sequence_num/frame_num.npy

import cv2
import numpy as np
import mediapipe as mp
import os
import time

# ─── CONFIG ──────────────────────────────────────────────────
SIGNS = [
    'hello',     # வணக்கம்
    'thank_you', # நன்றி
    'yes',       # ஆம்
    'no',        # இல்லை
    'sorry',     # மன்னிக்கவும்
    'please',    # தயவுசெய்து
    'love',      # அன்பு
    'help',      # உதவி
    'water',     # தண்ணீர்
    'good',      # நல்லது
    'bad',       # கெட்டது
    'eat',       # சாப்பிட
    'sleep',     # தூக்கம்
    'come',      # வாருங்கள்
    'go',        # போ
    'zero',      # பூஜ்யம்
    'one',       # ஒன்று
    'two',       # இரண்டு
    'three',     # மூன்று
    'four',      # நான்கு
    'five',      # ஐந்து
    'name',      # பெயர்
    'where',     # எங்கே
    'how',       # எப்படி
    'what',      # என்ன
]

NUM_SEQUENCES = 30    # 30 videos per sign
SEQUENCE_LENGTH = 30  # 30 frames per video
DATA_PATH = os.path.join('..', 'dataset')

# ─── MEDIAPIPE SETUP ─────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

# ─── CREATE DIRECTORY STRUCTURE ──────────────────────────────
def create_directories():
    for sign in SIGNS:
        for seq in range(NUM_SEQUENCES):
            path = os.path.join(DATA_PATH, sign, str(seq))
            os.makedirs(path, exist_ok=True)
    print(f"✅ Created directories for {len(SIGNS)} signs × {NUM_SEQUENCES} sequences")

# ─── EXTRACT KEYPOINTS FROM FRAME ────────────────────────────
def extract_keypoints(results):
    """
    Extract hand landmark keypoints from MediaPipe results.
    Returns flat array of (x, y, z) for each landmark.
    
    If 2 hands detected: 21 × 3 × 2 = 126 values
    If 1 hand detected:  21 × 3 = 63 values (padded to 126)
    """
    lh = np.zeros(21 * 3)  # Left hand (zeros if not detected)
    rh = np.zeros(21 * 3)  # Right hand (zeros if not detected)
    
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if i >= 2:
                break
            # Get hand label
            hand_label = results.multi_handedness[i].classification[0].label
            
            # Extract x, y, z for all 21 landmarks
            keypoints = np.array([
                [lm.x, lm.y, lm.z]
                for lm in hand_landmarks.landmark
            ]).flatten()
            
            if hand_label == 'Left':
                lh = keypoints
            else:
                rh = keypoints
    
    return np.concatenate([lh, rh])  # Total: 126 features per frame

# ─── DRAW ON FRAME ───────────────────────────────────────────
def draw_on_frame(frame, results, sign_name, sequence, frame_num):
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    # Status overlay
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (15, 23, 41), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    cv2.putText(frame, f'Sign: {sign_name.upper()}',
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 220, 160), 2)
    cv2.putText(frame, f'Sequence: {sequence+1}/{NUM_SEQUENCES}  Frame: {frame_num}/{SEQUENCE_LENGTH}',
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 200, 255), 1)
    
    # Progress bar
    progress = frame_num / SEQUENCE_LENGTH
    cv2.rectangle(frame, (10, 68), (w-10, 74), (50, 50, 80), -1)
    cv2.rectangle(frame, (10, 68), (int(10 + (w-20)*progress), 74), (100, 220, 160), -1)
    
    return frame

# ─── MAIN COLLECTION LOOP ────────────────────────────────────
def collect_data():
    create_directories()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "="*60)
    print("  SIGN LANGUAGE DATA COLLECTION")
    print("="*60)
    print(f"  Signs to collect: {len(SIGNS)}")
    print(f"  Sequences per sign: {NUM_SEQUENCES}")
    print(f"  Frames per sequence: {SEQUENCE_LENGTH}")
    print(f"  Total frames: {len(SIGNS) * NUM_SEQUENCES * SEQUENCE_LENGTH:,}")
    print("="*60)
    print("\n  CONTROLS:")
    print("  SPACE = Start recording sequence")
    print("  S     = Skip current sign")
    print("  Q     = Quit")
    print("="*60 + "\n")
    
    with hands:
        for sign_idx, sign in enumerate(SIGNS):
            print(f"\n[{sign_idx+1}/{len(SIGNS)}] Preparing: {sign.upper()}")
            
            # WAIT SCREEN — show sign name, wait for user ready
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                
                # Draw wait screen
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]),
                              (15, 23, 41), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                cv2.putText(frame, f'NEXT SIGN: {sign.upper()}',
                            (frame.shape[1]//2 - 150, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 220, 160), 2)
                cv2.putText(frame, f'Progress: {sign_idx}/{len(SIGNS)} signs done',
                            (frame.shape[1]//2 - 140, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 200, 255), 1)
                cv2.putText(frame, 'Press SPACE to start recording',
                            (frame.shape[1]//2 - 170, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, 'Press S to skip this sign',
                            (frame.shape[1]//2 - 130, 330),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
                
                cv2.imshow('Data Collection', frame)
                key = cv2.waitKey(10) & 0xFF
                if key == ord(' '):
                    break
                elif key == ord('s'):
                    print(f"  Skipped: {sign}")
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            
            # RECORD SEQUENCES
            for seq in range(NUM_SEQUENCES):
                # Countdown before each sequence
                for countdown in range(3, 0, -1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    cv2.putText(frame, str(countdown),
                                (frame.shape[1]//2 - 20, frame.shape[0]//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 4, (100, 220, 160), 6)
                    cv2.putText(frame, f'Sequence {seq+1}/{NUM_SEQUENCES}',
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imshow('Data Collection', frame)
                    cv2.waitKey(600)
                
                # Record SEQUENCE_LENGTH frames
                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)
                    
                    # Draw landmarks and UI
                    frame = draw_on_frame(frame, results, sign, seq, frame_num + 1)
                    cv2.imshow('Data Collection', frame)
                    cv2.waitKey(1)
                    
                    # Extract and save keypoints
                    keypoints = extract_keypoints(results)
                    save_path = os.path.join(DATA_PATH, sign, str(seq), str(frame_num))
                    np.save(save_path, keypoints)
                
                print(f"  ✅ Saved sequence {seq+1}/{NUM_SEQUENCES} for '{sign}'")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n" + "="*60)
    print("  ✅ DATA COLLECTION COMPLETE!")
    print(f"  Dataset saved to: {os.path.abspath(DATA_PATH)}")
    print("="*60)

if __name__ == '__main__':
    collect_data()
