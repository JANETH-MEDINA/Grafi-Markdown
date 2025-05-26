```python
import numpy as np
import numpy as np
import cv2 as cv
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv.VideoCapture(1)


ball_pos = np.array([320, 240], dtype=np.float32) 
ball_vel = np.array([5, 5], dtype=np.float32)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    x, y = frame.shape[:2]
    frame = cv.flip(frame, 1) 
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                hx, hy = int(landmark.x * y), int(landmark.y * x)

               
                if np.linalg.norm(ball_pos - np.array([hx, hy])) < 40:
                    ball_vel *= -1 

    ball_pos += ball_vel


    if ball_pos[0] <= 20 or ball_pos[0] >= y - 20:
        ball_vel[0] *= -1
    if ball_pos[1] <= 20 or ball_pos[1] >= x - 20:
        ball_vel[1] *= -1

    cv.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), 20, (0, 255, 0), -1)

    cv.rectangle(frame, (20, 20), (y-20, x-20), (234, 43, 34), 5)

    cv.imshow('Pelota en movimiento', frame)

    if cv.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
```