'''phyton
import numpy as np
import cv2 as cv

rostro = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

cap = cv.VideoCapture(1) 

if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el fotograma")
        break 

   
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

   
    rostros = rostro.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    count = len(rostros)  
    for (x, y, w, h) in rostros:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.putText(frame, f'Rostros detectados: {count}', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow('Rostros', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
'''

