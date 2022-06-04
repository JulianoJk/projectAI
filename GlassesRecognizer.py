import numpy as np
import cv2

face_with_glasses = cv2.CascadeClassifier('xmlFiles/haarcascade_eye_tree_eyeglasses.xml')
# image src
image = cv2.imread('images/maleSmileGlass.jpg')
# Gray out image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face = face_with_glasses.detectMultiScale(gray, 1.2, 5)


for (x, y, w, h) in face:
    image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = image[y:y + h, x:x + w]


cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
