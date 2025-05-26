```python
import cv2
import mediapipe as mp
import numpy as np
import random

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def reconocer_letra(hand_landmarks, frame):
    h, w, _ = frame.shape 
    dedos = [(int(hand_landmarks.landmark[i].x * w), int(hand_landmarks.landmark[i].y * h)) for i in range(21)]

    pulgar, indice, medio, anular, meñique, base_medio = dedos[4], dedos[8], dedos[12], dedos[16], dedos[20], dedos[9]

    for i, (x, y) in enumerate(dedos):
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(frame, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.putText(frame, f'({pulgar[0]}, {pulgar[1]})', (pulgar[0], pulgar[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (23, 0, 0), 2, cv2.LINE_AA)

    distancia_pulgar_indice = np.linalg.norm(np.array(pulgar) - np.array(indice))
    distancia_pulgar_medio = np.linalg.norm(np.array(pulgar) - np.array(base_medio))

    color = tuple(random.randint(0, 255) for _ in range(3))

    promedio = np.mean([np.linalg.norm(np.array(dedos[4]) - np.array(d)) for d in [dedos[8], dedos[12], dedos[16], dedos[20]]])
    centro_circulo = (pulgar[0] + 50, pulgar[1] + 50)

    if 0 < centro_circulo[0] < w and 0 < centro_circulo[1] < h:
        cv2.circle(frame, centro_circulo, int(promedio), color, -1)

    if distancia_pulgar_medio < 30:
        return "B"
    elif indice[1] < medio[1] < anular[1] < meñique[1]:
        return "A"
    elif distancia_pulgar_indice < 150:
        return "C"
    elif pulgar[1] > medio[1] and indice[1] < pulgar[1]:
        return "D"

    return "Desconocido"

cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            letra_detectada = reconocer_letra(hand_landmarks, frame)
            cv2.putText(frame, f"Letra: {letra_detectada}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Reconocimiento de Letras", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```