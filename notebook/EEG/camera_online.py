import cv2
import mediapipe as mp
import pyttsx3
import time
from collections import deque


TEXTS = {
    1: "Hello! This is the first predefined text to be read aloud.",
    2: "This is the second predefined text for demonstration purposes."
}


engine = pyttsx3.init()
engine.setProperty("rate", 160)

def speak(text):
    engine.stop()
    engine.say(text)
    engine.runAndWait()


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils


def fingers_up(hand_landmarks, handedness):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if handedness == "Right":
        fingers.append(
            1 if hand_landmarks.landmark[4].x <
                 hand_landmarks.landmark[3].x else 0
        )
    else:
        fingers.append(
            1 if hand_landmarks.landmark[4].x >
                 hand_landmarks.landmark[3].x else 0
        )

    # Other fingers
    for i in range(1, 5):
        fingers.append(
            1 if hand_landmarks.landmark[tips[i]].y <
                 hand_landmarks.landmark[tips[i] - 2].y else 0
        )

    return fingers


gesture_buffer = deque(maxlen=5)
last_action_time = 0
ACTION_COOLDOWN = 2.0  # seconds

current_action = "NONE"


cap = cv2.VideoCapture(0)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0].classification[0].label

        mp_draw.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )

        fingers = fingers_up(hand_landmarks, handedness)
        gesture_buffer.append(fingers)

        # Gesture stabilization
        if gesture_buffer.count(fingers) >= 4:
            finger_count = sum(fingers)
            now = time.time()

            if now - last_action_time > ACTION_COOLDOWN:

                # ✋ Open hand
                if finger_count == 5:
                    current_action = "IDLE"

                # ☝️ One finger → Read text 1
                elif finger_count == 1:
                    current_action = "READ TEXT 1"
                    speak(TEXTS[1])
                    last_action_time = now

                # ✌️ Two fingers → Read text 2
                elif finger_count == 2:
                    current_action = "READ TEXT 2"
                    speak(TEXTS[2])
                    last_action_time = now

                # 🤟 Three fingers → Custom action
                elif finger_count == 3:
                    current_action = "CUSTOM ACTION"
                    print("Custom action triggered!")
                    last_action_time = now

                # ✊ Fist → Stop speech
                elif finger_count == 0:
                    current_action = "STOP"
                    engine.stop()
                    last_action_time = now


    cv2.rectangle(frame, (10, 10), (420, 120), (0, 0, 0), -1)
    cv2.putText(frame, f"Action: {current_action}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.putText(frame, "Show fingers to trigger actions",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Gesture Controlled Actions", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
