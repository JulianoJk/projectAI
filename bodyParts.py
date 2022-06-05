import numpy as np
import cv2

lowerBody_cascade = cv2.CascadeClassifier('xmlFiles/haarcascade_lowerbody.xml')
upperBody_cascade = cv2.CascadeClassifier('xmlFiles/haarcascade_upperbody.xml')
fullBody_cascade = cv2.CascadeClassifier('xmlFiles/haarcascade_fullbody.xml')

# image src
img = cv2.imread('images/fullBody1.jpg')
# Gray out image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

lowerBody = lowerBody_cascade.detectMultiScale(gray, 1.05, 5)
upperBody = upperBody_cascade.detectMultiScale(gray, 1.05, 5)

# Detect and display upper body
for (x,y,w,h) in upperBody:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

# Detect and display lower body
for (x,y,w,h) in lowerBody:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()