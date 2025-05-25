```python
import cv2 as cv
import numpy as np

def cargar_imagen(ruta):
    """Carga la imagen en escala de grises y verifica si existe."""
    img = cv.imread(ruta, 0)
    if img is None:
        print("Error: No se pudo cargar la imagen.")
        exit()
    return img

def escalar_imagen(img, scale_x=2, scale_y=2):
    """Escala la imagen utilizando interpolación bicúbica."""
    return cv.resize(img, (img.shape[1] * scale_x, img.shape[0] * scale_y), interpolation=cv.INTER_CUBIC)

def rotar_imagen(img, angulo=45):
    """Rota la imagen manteniendo el tamaño correcto."""
    centro = (img.shape[1] // 2, img.shape[0] // 2)
    M = cv.getRotationMatrix2D(centro, angulo, 1.0)
    return cv.warpAffine(img, M, (img.shape[1], img.shape[0]))


imagen_original = cargar_imagen("OIP.jpeg")


imagen_escalada = escalar_imagen(imagen_original)
imagen_rotada = rotar_imagen(imagen_escalada)

cv.imshow('Imagen original', imagen_original)
cv.imshow('Imagen escalada', imagen_escalada)
cv.imshow('Imagen rotada', imagen_rotada)

cv.waitKey(0)
cv.destroyAllWindows()
```python