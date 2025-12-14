import cv2
import mediapipe as mp
import serial
import time

arduino = serial.Serial('/dev/ttyACM0', 9600)

time.sleep(2)

def rotate_servo(angle):
    arduino.write(f"{angle}\n".encode())
    print(f"Servo moved to {angle}°")


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

def fingers_up(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]  
    fingers = []


    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0]-1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for id in range(1, 5):
        if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id]-2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    
    return fingers


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            fingers = fingers_up(hand_landmarks)
            total_fingers = fingers.count(1)
            
            cv2.putText(frame, f'Fingers: {total_fingers}', (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            

            if total_fingers == 1:
                rotate_servo(0)
            elif total_fingers == 2:
                rotate_servo(30)
            elif total_fingers == 3:
                rotate_servo(90)
            elif total_fingers == 4:
                rotate_servo(150)
            elif total_fingers == 5:
                rotate_servo(180)

    cv2.imshow("Hand Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
