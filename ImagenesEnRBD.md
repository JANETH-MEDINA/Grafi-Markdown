```python
import cv2 as cv  
import numpy as np 

img = cv.imread('C:\\Users\\janet\\OneDrive\\Documentos\\TEC\\Grafi\\grafitareas\\OIP.jpeg', 1)
img2 = np.zeros((img.shape[:2]), dtype=np.uint8)

print(img.shape[:2])

r, g, b = cv.split(img)

img3 = cv.merge([r, b, g])

r = cv.merge([r, img2, img2])

g = cv.merge([img2, g, img2])

b = cv.merge([img2, img2, b])

cv.imshow('ejemplo', img)

cv.imshow('r', r)

cv.imshow('g', g)

cv.imshow('b', b)

cv.imshow('img3', img3)

cv.waitKey(0)

cv.destroyAllWindows()
```