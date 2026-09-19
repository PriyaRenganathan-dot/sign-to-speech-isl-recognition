from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

tip_ids = [4, 8, 12, 16, 20]

# -------- Tamil Font Load (Improved) --------

def put_tamil_text(img, text, position, color=(255,0,0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.truetype("C:/Windows/Fonts/Latha.ttf", 36)

    draw.text(position, text + "\u200c", font=font, fill=color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

num_text = {
    0: ("ZERO", "பூஜ்யம்"),
    1: ("ONE", "ஒன்று"),
    2: ("TWO", "இரண்டு"),
    3: ("THREE", "மூன்று"),
    4: ("FOUR", "நான்கு"),
    5: ("FIVE", "ஐந்து")
}

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w, _ = img.shape

    if result.multi_hand_landmarks and result.multi_handedness:

        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):

            lm_list = []
            for lm in hand_landmarks.landmark:
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            hand_label = result.multi_handedness[idx].classification[0].label

            finger_count = 0

            # Thumb
            if hand_label == "Right":
                if lm_list[4][0] > lm_list[3][0]:
                    finger_count += 1
            else:
                if lm_list[4][0] < lm_list[3][0]:
                    finger_count += 1

            # Other fingers
            for i in range(1, 5):
                if lm_list[tip_ids[i]][1] < lm_list[tip_ids[i] - 2][1]:
                    finger_count += 1

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            eng, tam = num_text[finger_count]

            x_pos = 30 if idx == 0 else w - 300

            cv2.putText(img, f"{hand_label} Hand", (x_pos, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            cv2.putText(img, f"{finger_count}", (x_pos, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            cv2.putText(img, f"{eng}", (x_pos, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # -------- Tamil Text --------
            img = put_tamil_text(img, tam, (x_pos, 160))

    cv2.imshow("Both Hands Finger to Text", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
