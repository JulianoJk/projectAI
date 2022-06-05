import numpy as np
import cv2


profileFace_cascade = cv2.CascadeClassifier('xmlFiles/haarcascade_profileface.xml')
# image src
img = cv2.imread('images/profileFace1.jpg')
# Gray out image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


for (x,y,w,h) in profileFace_cascade:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()