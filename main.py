import numpy as np
import cv2

# fbody_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# lbody_cascade = cv2.CascadeClassifier('haarcascade_lowerbody.xml')
ubody_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')
# fbody_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

img = cv2.imread('fullBody.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

body = ubody_cascade.detectMultiScale(gray, 1.05, 5)

for (x,y,w,h) in body:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()