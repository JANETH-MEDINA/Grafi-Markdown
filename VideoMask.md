'''phyton
import cv2
import numpy as np


cap = cv2.VideoCapture(1)

cv2.waitKey(2000)

ret, background = cap.read()
if not ret:
    print("No se puede capturar el fondo")
    cap.release()
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([110, 50, 0])
    upper_green = np.array([135, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    mask_inv = cv2.bitwise_not(mask)

    res1 = cv2.bitwise_and(frame, frame, mask=mask_inv)

    res2 = cv2.bitwise_and(background, background, mask=mask)


    final_output = cv2.addWeighted(res1, 1, res2, 1, 0) 

   
    cv2.imshow("invisible mask", final_output)
    cv2.imshow('mask', mask)

 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''