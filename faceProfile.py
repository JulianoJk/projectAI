import numpy as np
import cv2


face_cascade = cv2.CascadeClassifier('xmlFiles/haarcascade_profileface.xml')
# image src
image = cv2.imread('images/profileFace1.jpg')
# Gray out image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face = face_cascade.detectMultiScale(gray, 1.5, 5)

for (x, y, w, h) in face:
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = image[y:y + h, x:x + w]


cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
