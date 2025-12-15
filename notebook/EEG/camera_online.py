import cv2
import mediapipe as mp
import serial
import time
import math

arduino = serial.Serial('/dev/ttyACM0', 9600)
time.sleep(2)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def fingers_up(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    fingers.append(
        1 if hand_landmarks.landmark[4].x <
             hand_landmarks.landmark[3].x else 0
    )


    for i in range(1, 5):
        fingers.append(
            1 if hand_landmarks.landmark[tips_ids[i]].y <
                 hand_landmarks.landmark[tips_ids[i]-2].y else 0
        )
    return fingers

def finger_distance(lm1, lm2, w, h):
    x1, y1 = int(lm1.x * w), int(lm1.y * h)
    x2, y2 = int(lm2.x * w), int(lm2.y * h)
    return math.hypot(x2 - x1, y2 - y1)

def map_value(x, in_min, in_max, out_min, out_max):
    x = max(in_min, min(x, in_max))
    return int((x - in_min) * (out_max - out_min) /
               (in_max - in_min) + out_min)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            fingers = fingers_up(hand_landmarks)


            if fingers == [1, 1, 0, 0, 0]:
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                dist = finger_distance(thumb_tip, index_tip, w, h)

                speed = map_value(dist, 20, 200, 0, 255)
                arduino.write(f"{speed}\n".encode())

                cv2.putText(
                    frame,
                    f"Speed: {speed}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.line(
                    frame,
                    (int(thumb_tip.x * w), int(thumb_tip.y * h)),
                    (int(index_tip.x * w), int(index_tip.y * h)),
                    (255, 0, 0),
                    3
                )

    cv2.imshow("Gesture Speed Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()
