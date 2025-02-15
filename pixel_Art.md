```python
import cv2 as cv
import numpy as np
#cambios para commit
imagen= np.ones((425,425), dtype=np.uint8)*250
# for i in range(425):
#     for j in range (425):
#         if i%25==0 & j%25==0:
#             imagen [i,j]=1
#             imagen [j,i]=1 
            
            
for i in range(425):
    for j in range(425):
        if j<100 and j>75:
            if i>100 and i<175 or i>250 and i<325:
             imagen[j,i] = 1
        
        if j<125 and j>100:
            if i>75 and i<100 or i>175 and i<200 or  i>225 and i<250 or i>325 and i<350:
             imagen[j,i]=1
         
        if j<150 and j>125:
            if i>50 and i<75 or i>200 and i<225  or i>350 and i<375:
             imagen[j,i]=1 
                
        if j<175 and j>150:
            if i>50 and i<75  or i>350 and i<375:
             imagen[j,i]=1 
                    
        if j<200 and j>175:
            if i>50 and i<75  or i>350 and i<375:
             imagen[j,i]=1
                  
        if j<225 and j>200:
            if i>75 and i<100 or i>325 and i<350:
             imagen[j,i]=1  
             
        if j<250 and j>225:
            if i>100 and i<125 or i>300 and i<325:
             imagen[j,i]=1  
             
        if j<275 and j>250:
            if i>125 and i<150 or i>275 and i<300:
             imagen[j,i]=1  
         
        if j<300 and j>275:
            if i>150 and i<175 or i>250 and i<275:
             imagen[j,i]=1       
        
        if j<325 and j>300:
            if i>175 and i<200 or i>225 and i<250:
             imagen[j,i]=1 
        
        if j<350 and j>325:
            if i>200 and i<225:
             imagen[j,i]=1
             
        
        
      
cv.imshow('img',imagen)    
cv.waitKey(0)
cv.destroyAllWindows()
```