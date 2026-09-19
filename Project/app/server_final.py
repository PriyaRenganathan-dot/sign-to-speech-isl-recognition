# ============================================================
#  server_final.py — FINAL VERSION
#  Serves ISL_App_Final.html + WebSocket sign detection
# ============================================================
#  HOW TO RUN:
#  1. cd C:\Project
#  2. sign_env\Scripts\activate
#  3. cd app
#  4. python server_final.py
#  5. Open Chrome → http://localhost:8000
# ============================================================

import os, json, base64
import numpy as np
import cv2
import uvicorn
import mediapipe as mp

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# ── SIGN LIBRARY ─────────────────────────────────────────────
SIGNS = {
    'ZERO':      {'f':[0,0,0,0,0],'t':'பூஜ்யம்',         'tr':'Poojiyam'},
    'ONE':       {'f':[0,2,0,0,0],'t':'ஒன்று',           'tr':'Ondru'},
    'THREE':     {'f':[0,2,2,1,0],'t':'மூன்று',          'tr':'Moondru'},
    'FOUR':      {'f':[0,2,2,1,1],'t':'நான்கு',          'tr':'Naangu'},
    'FIVE':      {'f':[1,2,2,1,1],'t':'ஐந்து',           'tr':'Aindhu'},
    'SIX':       {'f':[1,0,0,0,1],'t':'ஆறு',             'tr':'Aaru'},
    'SEVEN':     {'f':[1,2,0,0,1],'t':'ஏழு',             'tr':'Aezhu'},
    'NINE':      {'f':[1,0,0,1,1],'t':'ஒன்பது',          'tr':'Onbadu'},
    'HELLO':     {'f':[1,2,0,1,0],'t':'வணக்கம்',         'tr':'Vanakkam'},
    'GOODBYE':   {'f':[1,0,1,0,1],'t':'விடைபெறுகிறேன்', 'tr':'Vidaiperu'},
    'THANKS':    {'f':[1,0,0,1,0],'t':'நன்றி',           'tr':'Nandri'},
    'SORRY':     {'f':[0,2,0,1,1],'t':'மன்னிக்கவும்',    'tr':'Mannikkavum'},
    'PLEASE':    {'f':[1,2,2,0,0],'t':'தயவுசெய்து',      'tr':'Thayavuseithu'},
    'LOVE':      {'f':[1,2,0,0,0],'t':'அன்பு',           'tr':'Anbu'},
    'HELP':      {'f':[0,2,0,0,1],'t':'உதவி',            'tr':'Udavi'},
    'YES':       {'f':[1,0,0,0,0],'t':'ஆம்',             'tr':'Aam'},
    'NO':        {'f':[0,0,0,0,1],'t':'இல்லை',           'tr':'Illai'},
    'GOOD':      {'f':[0,2,0,1,0],'t':'நல்லது',          'tr':'Nalladu'},
    'BAD':       {'f':[0,0,1,0,1],'t':'கெட்டது',         'tr':'Kettadu'},
    'STOP':      {'f':[0,0,1,1,1],'t':'நிறுத்து',        'tr':'Niruthu'},
    'GO':        {'f':[1,2,0,1,1],'t':'போங்கள்',         'tr':'Pogal'},
    'WATER':     {'f':[1,0,1,0,0],'t':'தண்ணீர்',         'tr':'Thanneer'},
    'DRINK':     {'f':[0,0,1,0,0],'t':'குடிக்க',         'tr':'Kudikka'},
    'RICE':      {'f':[0,0,1,1,0],'t':'சோறு',            'tr':'Soru'},
    'PAIN':      {'f':[0,0,0,1,1],'t':'வலி',             'tr':'Vali'},
    'DOCTOR':    {'f':[1,0,1,1,1],'t':'டாக்டர்',         'tr':'Doctor'},
    'WHERE':     {'f':[1,1,0,0,0],'t':'எங்கே',           'tr':'Enge'}
}

# ── MEDIAPIPE ─────────────────────────────────────────────────
print("  Loading MediaPipe...")
mp_hands   = mp.solutions.hands
hands_proc = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.80,
    min_tracking_confidence=0.75,
)
print("  MediaPipe ready!")

# ── FIND HTML FILE ────────────────────────────────────────────
SEARCH_DIRS = [
    r"C:\Project",
    os.path.join(os.path.expanduser("~"), "Project"),
    os.path.join(os.path.expanduser("~"), "Downloads"),
    os.path.dirname(os.path.abspath(__file__)),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
]
HTML_NAMES = ["ISL_App_Final.html","ISL_App_v4.html","ISL_App_v3.html","ISL_App_v2.html"]

HTML_FILE = None
for d in SEARCH_DIRS:
    for name in HTML_NAMES:
        candidate = os.path.join(d, name)
        if os.path.exists(candidate):
            HTML_FILE = candidate
            print(f"  Found HTML: {HTML_FILE}")
            break
    if HTML_FILE:
        break

if not HTML_FILE:
    print("  WARNING: HTML not found! Put ISL_App_Final.html in C:\\Project\\")

# ── HELPERS ───────────────────────────────────────────────────
def get_fingers(lm, hand):
    f = []
    f.append(1 if (hand=='Right' and lm[4].x < lm[3].x) or
                  (hand=='Left'  and lm[4].x > lm[3].x) else 0)
    iy, pip, dip = lm[8].y, lm[6].y, lm[7].y
    if iy < pip:   f.append(2)
    elif iy < dip: f.append(1)
    else:          f.append(0)
    for tip, pip_j in [(12,10),(16,14),(20,18)]:
        f.append(1 if lm[tip].y < lm[pip_j].y else 0)
    return f

def match_sign(fingers):
    for name, data in SIGNS.items():
        if fingers == data['f']:
            return name, data['t'], data['tr'], 1.0
    best, bt, btr, bs = None,'','',0
    for name, data in SIGNS.items():
        sc = sum(a==b for a,b in zip(fingers, data['f'])) / 5
        if sc > bs: best,bt,btr,bs = name,data['t'],data['tr'],sc
    return (best,bt,btr,bs) if bs >= 0.8 else (None,'','',0.0)

# ── FASTAPI ───────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_origin_regex=".*", allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def home():
    if HTML_FILE and os.path.exists(HTML_FILE):
        with open(HTML_FILE, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>HTML not found — put ISL_App_Final.html in C:\\Project\\</h1>")

@app.get("/status")
async def status():
    return {"status":"running","signs":len(SIGNS),"html":str(HTML_FILE)}

# ── WEBSOCKET ─────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("  Browser connected!")
    last_sign, stable_cnt, STABLE = '', 0, 18
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            if msg.get('type') == 'ping':
                await ws.send_text(json.dumps({'type':'pong'}))
                continue
            if msg.get('type') == 'frame':
                try:
                    b64   = msg['frame'].split(',')[-1]
                    arr   = np.frombuffer(base64.b64decode(b64), np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None: continue
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = hands_proc.process(rgb)
                    sign=tamil=tr=None; conf=0.0
                    if res.multi_hand_landmarks:
                        hlm   = res.multi_hand_landmarks[0]
                        label = res.multi_handedness[0].classification[0].label
                        fi    = get_fingers(hlm.landmark, label)
                        sign, tamil, tr, conf = match_sign(fi)
                    if sign:
                        if sign==last_sign: stable_cnt+=1
                        else: last_sign=sign; stable_cnt=1
                        confirmed = stable_cnt >= STABLE
                        if confirmed: stable_cnt=0
                        await ws.send_text(json.dumps({
                            'detected':True,'sign_english':sign,
                            'sign_tamil':tamil,'sign_tr':tr,
                            'confidence':round(conf*100),
                            'confirmed':confirmed,
                            'confirmed_sign':sign if confirmed else '',
                            'confirmed_tamil':tamil if confirmed else '',
                        }))
                    else:
                        stable_cnt=max(0,stable_cnt-1)
                        await ws.send_text(json.dumps({'detected':False}))
                except Exception:
                    continue
    except WebSocketDisconnect:
        print("  Browser disconnected")
    except Exception as e:
        print(f"  Error: {e}")

# ── RUN ───────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n"+"="*52)
    print("  ISL SIGN RECOGNITION — SERVER FINAL")
    print("="*52)
    print(f"  HTML  : {'FOUND ✅' if HTML_FILE else 'NOT FOUND ❌'}")
    if HTML_FILE: print(f"  File  : {os.path.basename(HTML_FILE)}")
    print(f"  Signs : {len(SIGNS)}")
    print("="*52)
    print("\n  Open Chrome → http://localhost:8000")
    print("  Press Ctrl+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=8000,
                ws_ping_interval=20, ws_ping_timeout=20)
