import cv2
import mediapipe as mp
import time
from collections import deque

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Smoothing buffer
gesture_buffer = deque(maxlen=8)

# Wave detection
prev_x = None
wave_counter = 0

# Finger tips
FINGER_TIPS = [4, 8, 12, 16, 20]

def fingers_up(lm):
    fingers = []

    # Thumb (x-axis)
    fingers.append(1 if lm[4].x < lm[3].x else 0)

    # Other fingers (y-axis)
    for tip in [8, 12, 16, 20]:
        fingers.append(1 if lm[tip].y < lm[tip - 2].y else 0)

    return fingers

def smooth_gesture(gesture):
    gesture_buffer.append(gesture)
    return max(set(gesture_buffer), key=gesture_buffer.count)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "No Gesture"

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        fingers = fingers_up(lm)
        count = fingers.count(1)

        palm_x = lm[9].x

        # 👋 Wave
        if count == 5:
            if prev_x is not None and abs(palm_x - prev_x) > 0.04:
                wave_counter += 1
            prev_x = palm_x

            if wave_counter > 6:
                gesture = "WAVE 👋"
                wave_counter = 0
            else:
                gesture = "OPEN PALM ✋"

        # ✊ Fist
        elif count == 0:
            gesture = "FIST ✊"

        # ☝️ Point
        elif fingers == [0, 1, 0, 0, 0]:
            gesture = "POINT ☝️"

        # 👍 Thumbs up
        elif fingers == [1, 0, 0, 0, 0]:
            gesture = "THUMBS UP 👍"

        # ✌️ Victory
        elif fingers == [0, 1, 1, 0, 0]:
            gesture = "VICTORY ✌️"

        # 👌 OK
        elif fingers == [1, 1, 0, 0, 0]:
            gesture = "OK 👌"
                # 🔢 THREE fingers
        elif fingers == [0, 1, 1, 1, 0]:
            gesture = "THREE 3️⃣"

        # 🔢 FOUR fingers
        elif fingers == [0, 1, 1, 1, 1]:
            gesture = "FOUR 4️⃣"

        else:
            gesture = "UNKNOWN"

        gesture = smooth_gesture(gesture)
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Gesture: {gesture}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Advanced Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
