'''phyton
import cv2 as cv
import numpy as np

img = cv.imread('OIP.jpeg', cv.IMREAD_GRAYSCALE)


if img is None:
    print("Error: No se pudo cargar la imagen.")
    exit()


scale_img = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)


kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16
convol_img = cv.filter2D(scale_img, -1, kernel)


equalized_img = cv.equalizeHist(convol_img)


cv.imshow('Imagen original', img)
cv.imshow('Imagen escalada', scale_img)
cv.imshow('Imagen convolucionada', convol_img)
cv.imshow('Imagen con contraste mejorado', equalized_img)
cv.waitKey(0)
cv.destroyAllWindows()
'''