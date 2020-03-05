# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:18:35 2020

@author: cea6abt
"""

# Face Detection from a list of images

import cv2
import os
import glob
import numpy as np

#%% storing images face detection
images = []
path = "C:\\Users\\CEA6ABT\\Documents\\resources\\ComputerVision\\Computer_Vision_A_Z_Template_Folder\\images\\*.*"
for file in glob.glob(path):
    print(file)
    a= cv2.imread(file)
    images.append(a)
#%% OpenCV face detection
    

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # We load the cascade for the face.
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') # We load the cascade for the eyes.
    

for i in images:
    gray = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5) # We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    for (x, y, w, h) in faces: # For each detected face:
        gray = cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 0, 0), 2) # We paint a rectangle around the face.
        roi_gray = gray[y:y+h, x:x+w] # We get the region of interest in the black and white image.
        roi_color = gray[y:y+h, x:x+w] # We get the region of interest in the colored image.
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=7) # We apply the detectMultiScale method to locate one or several eyes in the image.
        for (ex, ey, ew, eh) in eyes: # For each detected eye:
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2) # We paint a rectangle around the eyes, but inside the referential of the face.
    cv2.imshow('img', gray)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()  
  

 # We destroy all the windows inside which the images were displayed.