```python
import cv2
import mediapipe as mp
import numpy as np

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def distancia(p1, p2):
    return np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

def detectar_gesto(landmarks):
    # Puntos de los dedos
    pulgar_tip = landmarks.landmark[4]
    indice_tip = landmarks.landmark[8]
    medio_tip = landmarks.landmark[12]
    anular_tip = landmarks.landmark[16]
    menique_tip = landmarks.landmark[20]

    # Bases
    base_indice = landmarks.landmark[5]
    base_medio = landmarks.landmark[9]
    base_anular = landmarks.landmark[13]
    base_menique = landmarks.landmark[17]
    base_pulgar = landmarks.landmark[2]

    # Medias
    indice_pip = landmarks.landmark[6]
    medio_pip = landmarks.landmark[10]
    anular_pip = landmarks.landmark[14]
    menique_pip = landmarks.landmark[18]

    # Distancias
    d_pulgar_baseindice = distancia(pulgar_tip, base_indice)
    d_pulgar_indice = distancia(pulgar_tip, indice_tip)
    d_pulgar_medio = distancia(pulgar_tip, medio_tip)
    d_pulgar_anular = distancia(pulgar_tip, anular_tip)
    d_pulgar_menique = distancia(pulgar_tip, menique_tip)
    d_indice_medio = distancia(indice_tip, medio_tip)
    d_indice_base = distancia(indice_tip, base_indice)
    d_baseanular_pulgar = distancia(base_anular, pulgar_tip)
    d_medio_base = distancia(medio_tip, base_medio)
    d_anular_base = distancia(anular_tip, base_anular)
    d_menique_base = distancia(menique_tip, base_menique)
    d_anular_menique = distancia(anular_tip, menique_tip)
    d_pulgar_base = distancia(pulgar_tip, base_pulgar)
    d_indice_pip = distancia(indice_tip, indice_pip)
    d_medio_pip = distancia(medio_tip, medio_pip)
    d_anular_pip = distancia(anular_tip, anular_pip)
    d_menique_pip = distancia(menique_tip, menique_pip)

    # Gestos: F, P, Y, 10, 19, 33, Amigo
    
    # F - Índice y medio 
    if (d_indice_base > 0.15 and d_medio_base > 0.15 and 
        d_anular_base < 0.10 and d_menique_base < 0.10 and
        d_pulgar_anular < 0.08):
        return "F"
    
    # P - Solo índice
    elif (d_indice_base > 0.16 and d_medio_base < 0.10 and 
          d_anular_base < 0.10 and d_menique_base < 0.10 and
          d_pulgar_indice > 0.10 and d_pulgar_medio < 0.08):
        return "P"
    
    # Y - Pulgar y meñique 
    elif (d_pulgar_base > 0.12 and d_menique_base > 0.15 and
          d_indice_base < 0.10 and d_medio_base < 0.10 and d_anular_base < 0.10):
        return "Y"
    
    # 10 - Índice extendido hacia el pulgar
    elif (d_pulgar_base > 0.13 and d_indice_base > 0.15 and
          d_pulgar_indice < 0.12 and d_medio_base < 0.10 and
          d_anular_base < 0.10 and d_menique_base < 0.10):
        return "10"
    
    # 19 - Índice y meñique extendidos
    elif (d_indice_base > 0.15 and d_menique_base > 0.15 and
          d_medio_base < 0.10 and d_anular_base < 0.10 and
          d_pulgar_indice > 0.12 and d_pulgar_menique > 0.15):
        return "19"
    
    # 33 - Índice, medio y anular
    elif (d_indice_base > 0.15 and d_medio_base > 0.15 and d_anular_base > 0.15 and
          d_menique_base < 0.12 and (d_pulgar_base < 0.12 or d_pulgar_indice < 0.12)):
        return "33"
    
    # Amigo - Mano abierta
    elif (d_indice_base > 0.15 and d_medio_base > 0.15 and 
          d_anular_base > 0.15 and d_menique_base > 0.15 and
          d_pulgar_base > 0.12):
        return "Amigo"
    
    else:
        return "No detectado"

# Captura de video
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        # Mostrar gestos individuales únicamente
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            etiqueta_mano = handedness.classification[0].label 
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesto = detectar_gesto(hand_landmarks)
            
            if gesto and gesto != "No detectado":
                # Mostrar el gesto detectado en pantalla
                cv2.putText(frame, f" {gesto}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)

    cv2.imshow("Deteccion de Gestos LSM - F, P, Y, 10, 19, 33, Amigo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```