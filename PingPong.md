```python
import cv2 as cv
import numpy as np

x=0
y=0

x2=5 
y2=2
while True:
    img = np.ones((500,500,3),dtype=np.uint8) *255
    
    x+=x2
    y+=y2
    cv.circle(img, (x,y), 20, (160,150,50), -1)
    
    if x <= 0 or x >= 500:
        x2=-x2
    if y <= 0 or y >= 500:
        y2=-y2
    
    cv.imshow('img',img)
    cv.waitKey(1)

cv.destroyAllWindows()
```