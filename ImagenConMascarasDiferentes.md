'''phyton
import cv2 as cv
import numpy as np

img = cv.imread('OIP.jpeg', cv.IMREAD_COLOR)

if img is None:
    print("Error: No se pudo cargar la imagen.")
    exit()

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

uba = (10, 255, 255)   
ubb = (0, 50, 50)     

uba2 = (179, 255, 255)  
ubb2 = (170, 50, 50)    

uba3 = (110, 255, 255)
ubb3 = (90, 50, 50)

uba4 = (40, 255, 255)
ubb4 = (20, 100, 100)


mask1 = cv.inRange(hsv, ubb, uba)
mask2 = cv.inRange(hsv, ubb2, uba2)
mask3 = cv.inRange(hsv, ubb3, uba3)
mask4 = cv.inRange(hsv, ubb4, uba4)

mask = cv.bitwise_or(mask1, mask2)
mask = cv.bitwise_or(mask, mask3)
mask = cv.bitwise_or(mask, mask4)


res = cv.bitwise_and(img, img, mask=mask)

cv.imshow('Máscara Rojo', mask1)
cv.imshow('Máscara Rojo Claro', mask2)
cv.imshow('Máscara Azul', mask3)
cv.imshow('Máscara Verde', mask4)
cv.imshow('Máscara Final', mask)
cv.imshow('Imagen Filtrada', res)

cv.waitKey(0) & 0xFF
cv.destroyAllWindows()
'''