# ============================================================
#  ISL SIGN RECOGNITION — Pre-trained Mode  VERSION 2
#  Tamil text FIXED using Pillow!
#  No dataset needed! No training needed!
# ============================================================
#  HOW TO RUN:
#  1. cd C:\Project
#  2. sign_env\Scripts\activate
#  3. pip install Pillow
#  4. cd app
#  5. python pretrained_predict_v2.py

import cv2
import numpy as np
import mediapipe as mp
import time
import threading
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import os
import urllib.request

# ── DOWNLOAD TAMIL FONT (one time only) ──────────────────────
FONT_PATH = "NotoSansTamil.ttf"

def download_tamil_font():
    if os.path.exists(FONT_PATH):
        print("  Tamil font found!")
        return True
    print("  Downloading Tamil font (one time, ~200KB)...")
    url = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansTamil/NotoSansTamil-Regular.ttf"
    try:
        urllib.request.urlretrieve(url, FONT_PATH)
        print("  Tamil font downloaded!")
        return True
    except Exception as e:
        print(f"  Font download failed: {e}")
        print("  Will show English transliteration instead")
        return False

TAMIL_FONT_OK = download_tamil_font()
try:
    font_large = ImageFont.truetype(FONT_PATH, 38) if TAMIL_FONT_OK else None
    font_small = ImageFont.truetype(FONT_PATH, 22) if TAMIL_FONT_OK else None
except:
    font_large = font_small = None
    TAMIL_FONT_OK = False

# ── MEDIAPIPE ────────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_draw_sty = mp.solutions.drawing_styles
hands = mp_hands.Hands(max_num_hands=2,
    min_detection_confidence=0.75, min_tracking_confidence=0.70)

# ── SIGN LIBRARY (30 signs) ──────────────────────────────────
# f = finger pattern [thumb,index,middle,ring,pinky] 1=up 0=down
# t = Tamil text, tr = English transliteration
SIGNS = {
   'ZERO':     {f:[0,0,0,0,0],ta:'பூஜ்யம்',   tr:'Poojiyam'},
  'ONE':      {f:[0,2,0,0,0],ta:'ஒன்று',     tr:'Ondru'},
  'THREE':    {f:[0,2,2,1,0],ta:'மூன்று',    tr:'Moondru'},
  'FOUR':     {f:[0,2,2,1,1],ta:'நான்கு',    tr:'Naangu'},
  'FIVE':     {f:[1,2,2,1,1],ta:'ஐந்து',     tr:'Aindhu'},
  'SIX':      {f:[1,0,0,0,1],ta:'ஆறு',       tr:'Aaru'},
  'SEVEN':    {f:[1,2,0,0,1],ta:'ஏழு',       tr:'Aezhu'},
  'NINE':     {f:[1,0,0,1,1],ta:'ஒன்பது',    tr:'Onbadu'},
  'HELLO':    {f:[1,2,0,1,0],ta:'வணக்கம்',   tr:'Vanakkam'},
  'GOODBYE':  {f:[1,0,1,0,1],ta:'விடைபெறுகிறேன்',tr:'Vidaiperu'},
  'THANKS':   {f:[1,0,0,1,0],ta:'நன்றி',     tr:'Nandri'},
  'SORRY':    {f:[0,2,0,1,1],ta:'மன்னிக்கவும்',tr:'Mannikkavum'},
  'PLEASE':   {f:[1,2,2,0,0],ta:'தயவுசெய்து',tr:'Thayavuseithu'},
  'LOVE':     {f:[1,2,0,0,0],ta:'அன்பு',     tr:'Anbu'},
  'HELP':     {f:[0,2,0,0,1],ta:'உதவி',      tr:'Udavi'},
  'YES':      {f:[1,0,0,0,0],ta:'ஆம்',       tr:'Aam'},
  'NO':       {f:[0,0,0,0,1],ta:'இல்லை',     tr:'Illai'},
  'GOOD':     {f:[0,2,0,1,0],ta:'நல்லது',    tr:'Nalladu'},
  'BAD':      {f:[0,0,1,0,1],ta:'கெட்டது',   tr:'Kettadu'},
  'STOP':     {f:[0,0,1,1,1],ta:'நிறுத்து',  tr:'Niruthu'},
  'GO':       {f:[1,2,0,1,1],ta:'போங்கள்',   tr:'Pogal'},
  'WATER':    {F:[1,0,1,0,0],ta:'தண்ணீர்',         tr:'Thanneer'},
  'DRINK':    {f:[0,0,1,0,0],ta:'குடிக்க',   tr:'Kudikka'},
  'RICE':     {f:[0,0,1,1,0],ta:'சோறு',            tr:'Soru'},
  'PAIN':     {f:[0,0,0,1,1],ta:'வலி',       tr:'Vali'},
  'DOCTOR':   {f:[1,0,1,1,1],ta:'டாக்டர்',   tr:'Doctor'},
  'WHERE':    {f:[1,1,0,0,0],ta:'எங்கே',     tr:'Enge'}
}

# ── TTS VOICE ────────────────────────────────────────────────
TTS_OK = False
tts = None
try:
    import pyttsx3
    tts = pyttsx3.init()
    tts.setProperty('rate', 175)
    tts.setProperty('volume', 1.0)
    for v in tts.getProperty('voices'):
        if 'female' in v.name.lower() or 'zira' in v.name.lower():
            tts.setProperty('voice', v.id)
            break
    TTS_OK = True
    print("  Voice TTS ready!")
except:
    print("  TTS not available — visual only")

def speak(text):
    if not TTS_OK: return
    def _s():
        try: tts.say(text); tts.runAndWait()
        except: pass
    threading.Thread(target=_s, daemon=True).start()

# ── TAMIL TEXT USING PILLOW ───────────────────────────────────
def put_tamil(frame, text, x, y, font, color=(255,220,60)):
    """Render Tamil Unicode text onto OpenCV frame using Pillow."""
    if font is None:
        return frame
    try:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(pil).text((x,y), text, font=font, fill=color)
        frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except:
        pass
    return frame

# ── FINGER STATE DETECTION ───────────────────────────────────
def get_fingers(lm, hand):
    f = []
    f.append(1 if (hand=='Right' and lm[4].x < lm[3].x) or
                  (hand=='Left'  and lm[4].x > lm[3].x) else 0)
    for tip,pip in [(8,6),(12,10),(16,14),(20,18)]:
        f.append(1 if lm[tip].y < lm[pip].y else 0)
    return f

# ── SIGN MATCHING ────────────────────────────────────────────
def match(fingers):
    best, bt, btr, bs = None,'','',0
    for name,data in SIGNS.items():
        s = sum(a==b for a,b in zip(fingers,data['f'])) / 5
        if s > bs: best,bt,btr,bs = name,data['t'],data['tr'],s
    return (best,bt,btr,bs) if bs >= 0.8 else (None,'','',0.0)

# ── DRAW HUD ─────────────────────────────────────────────────
def hud(frame, sign, tamil, tr, conf, fingers, sentence, fps, nh):
    H,W = frame.shape[:2]

    # Top bar
    ov = frame.copy()
    cv2.rectangle(ov,(0,0),(W,100),(8,12,25),-1)
    cv2.addWeighted(ov,0.83,frame,0.17,0,frame)

    cv2.putText(frame,'ISL SIGN RECOGNITION  v2',
        (10,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(100,200,255),2)
    cv2.putText(frame,f'FPS:{fps:.0f}  Hands:{nh}',
        (W-165,28),cv2.FONT_HERSHEY_SIMPLEX,0.55,(140,220,140),1)

    # Finger dots T I M R P
    for i,(nm,st) in enumerate(zip(['T','I','M','R','P'],fingers)):
        c = (70,220,110) if st else (55,55,95)
        cv2.circle(frame,(W-152+i*30,65),13,c,-1)
        cv2.putText(frame,nm,(W-157+i*30,70),
            cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)

    # English sign large
    if sign:
        col = (70,220,110) if conf>=1.0 else (70,195,255)
        cv2.putText(frame,sign,(10,72),
            cv2.FONT_HERSHEY_SIMPLEX,1.35,col,3)
        bw = int((W-180)*conf)
        cv2.rectangle(frame,(10,83),(W-170,91),(35,45,75),-1)
        cv2.rectangle(frame,(10,83),(10+bw,91),col,-1)
        cv2.putText(frame,f'{conf*100:.0f}%',
            (W-158,91),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1)

    # Bottom bar
    ov2 = frame.copy()
    cv2.rectangle(ov2,(0,H-115),(W,H),(8,12,25),-1)
    cv2.addWeighted(ov2,0.80,frame,0.20,0,frame)

    # Tamil label
    if tamil:
        cv2.putText(frame,'Tamil:',
            (10,H-82),cv2.FONT_HERSHEY_SIMPLEX,0.6,(180,180,200),1)
        if TAMIL_FONT_OK and font_large:
            # Proper Tamil script via Pillow
            frame = put_tamil(frame, tamil, 85, H-103,
                              font_large, (255,220,60))
        else:
            # Fallback: show transliteration
            cv2.putText(frame, tr,
                (85,H-82),cv2.FONT_HERSHEY_SIMPLEX,
                0.75,(255,220,60),2)

    # Sentence
    if sentence:
        cv2.putText(frame,'Sentence:',
            (10,H-48),cv2.FONT_HERSHEY_SIMPLEX,0.5,(160,180,210),1)
        cv2.putText(frame,'  |  '.join(list(sentence)[-5:]),
            (105,H-48),cv2.FONT_HERSHEY_SIMPLEX,0.52,(180,225,255),1)

    cv2.putText(frame,'Q=Quit  C=Clear  SPACE=Speak  S=Save',
        (10,H-16),cv2.FONT_HERSHEY_SIMPLEX,0.44,(90,110,155),1)
    return frame

# ── MAIN ─────────────────────────────────────────────────────
def run():
    print("\n" + "="*55)
    print("  ISL SIGN RECOGNITION v2 — Tamil Text Fixed!")
    print(f"  Signs: {len(SIGNS)}")
    print(f"  Tamil: {'Proper script' if TAMIL_FONT_OK else 'English fallback'}")
    print(f"  Voice: {'Ready' if TTS_OK else 'Not available'}")
    print("="*55)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    if not cap.isOpened():
        print("Camera not found!"); return

    sentence = deque(maxlen=20)
    last_sign = ''; stable = 0; NEED = 20
    cs=ct=ctr=''; cc=0.0
    fingers=[0]*5; fps_t=time.time(); fc=0; fps=0
    history=[]

    with hands:
        while True:
            ret,frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame,1)
            rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            rgb.flags.writeable=False
            res = hands.process(rgb)
            rgb.flags.writeable=True

            nh=0; ds=dt=dtr=''; dc=0.0

            if res.multi_hand_landmarks:
                nh = len(res.multi_hand_landmarks)
                for hlm in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame,hlm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_draw_sty.get_default_hand_landmarks_style(),
                        mp_draw_sty.get_default_hand_connections_style())

                hlm   = res.multi_hand_landmarks[0]
                label = res.multi_handedness[0].classification[0].label
                fingers = get_fingers(hlm.landmark,label)
                sign,tamil,tr,conf = match(fingers)

                if sign:
                    ds,dt,dtr,dc = sign,tamil,tr,conf
                    stable = stable+1 if sign==last_sign else 1
                    last_sign = sign
                    if stable==NEED:
                        cs,ct,ctr,cc = sign,tamil,tr,conf
                        print(f"  {sign} | {tamil} ({tr}) | {conf*100:.0f}%")
                        if not sentence or sentence[-1]!=sign:
                            sentence.append(sign); speak(sign)
                else:
                    stable = max(0,stable-1)
            else:
                fingers=[0]*5; stable=max(0,stable-2)

            fc+=1
            el=time.time()-fps_t
            if el>=1.0: fps=fc/el; fc=0; fps_t=time.time()

            s=ds or cs; t=dt or ct; tr2=dtr or ctr
            c=dc if ds else cc*0.5
            frame = hud(frame,s,t,tr2,c,fingers,sentence,fps,nh)
            cv2.imshow('ISL Sign Recognition v2 — Tamil Fixed!',frame)

            k=cv2.waitKey(1)&0xFF
            if k==ord('q'): break
            elif k==ord('c'):
                sentence.clear(); cs=ct=ctr=''; cc=0
                print("  Cleared")
            elif k==ord(' ') and sentence:
                full=' '.join(sentence); speak(full)
                print(f"  Speaking: {full}")
            elif k==ord('s') and cs:
                e=f"{cs} ({ct})"; history.append(e)
                print(f"  Saved: {e}")

    cap.release(); cv2.destroyAllWindows()
    if history: print(f"\n  History: {history}")
    print("\nDone! All the best bro! 🤟🎓")

if __name__=='__main__': run()
