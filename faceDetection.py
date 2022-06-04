import numpy as np
import cv2
import os, sys


# to specify folder path (set working directory)
os.chdir('add path here')

# to get working directory use the line bellow (folder must be opened in editor)
#os.getcwd()

#comment the line bellow for mac!
#uncomment to print full lenght on nd.array
#np.set_printoptions(threshold='nan')


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('face.jpg')

#comment the line bellow to and run again to see the image recognition
cv2.imshow('photo1',img)

#convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('photo2',gray)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
print(x,y,w,h)

#uncomment to see the classification
#cv2.imshow('photo3',img)

# To show both images in one frame
# comment lines  14, 18 , 28
# Make the grey scale image have three channels
#gray_3_channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
#matrixHorizontal=np.hstack((img, gray_3_channel))
#matrixHorizontal_concat= np.concatenate((img, gray_3_channel), axis=1)
#cv2.imshow('Numpy Horizontal Concat', matrixHorizontal_concat)

cv2.waitKey(0)
cv2.destroyAllWindows()
