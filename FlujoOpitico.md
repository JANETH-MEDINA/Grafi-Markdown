```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(1)  

lkparm = dict(winSize=(15,15), maxLevel=2,
              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

ret, vframe = cap.read()
if not ret:
    print("Error al capturar el frame inicial.")
    cap.release()
    exit()

vgris = cv.cvtColor(vframe, cv.COLOR_BGR2GRAY)

# Detectamos características en la imagen en lugar de puntos manuales
p0 = cv.goodFeaturesToTrack(vgris, maxCorners=100, qualityLevel=0.3, minDistance=7)
p0 = np.float32(p0)

mask = np.zeros_like(vframe)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgris = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    p1, st, err = cv.calcOpticalFlowPyrLK(vgris, fgris, p0, None, **lkparm)

    if p1 is None:
        vgris = fgris.copy()
        p0 = cv.goodFeaturesToTrack(vgris, maxCorners=100, qualityLevel=0.3, minDistance=7)
        p0 = np.float32(p0)
        mask = np.zeros_like(frame)
    else:
        bp1 = p1[st == 1]
        bp0 = p0[st == 1]

        for nv, vj in zip(bp1, bp0):
            a, b = int(nv[0]), int(nv[1])
            c, d = int(vj[0]), int(vj[1])
            frame = cv.circle(frame, (a, b), 3, (0, 255, 0), -1)
            frame = cv.circle(frame, (c, d), 2, (255, 0, 0), -1)
            frame = cv.line(frame, (a, b), (c, d), (0, 0, 255), 1)

        cv.imshow('Flujo óptico', frame)
        vgris = fgris.copy()
        
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
```