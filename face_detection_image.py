# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:18:35 2020

@author: cea6abt
"""

# Face Detection from a list of images

import cv2
#from os.path import isfile, join
#from os import listdir
import os
import glob
import numpy as np

# Get all images from the directory and append them to a list.
# mypath='C:/Users/CEA6ABT/Documents/resources/ComputerVision/Computer_Vision_A_Z_Template_Folder/images'
# onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
# images = np.empty(len(onlyfiles), dtype=object)
# grays = np.empty(len(onlyfiles), dtype=object)
# for n in range(0, len(onlyfiles)):
#     images[n] = cv2.imread( join(mypath,onlyfiles[n]) )
# for n in images:q
#     grays[n] = cv2.cvtColor(images[n], cv2.COLOR_BGR2GRAY)    

absolute_path = os.path.join('..', 'images', 'photo-fr-2.jpeg');
  
img = cv2.imread(absolute_path)
cv2.imshow('img', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # We load the cascade for the face.
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') # We load the cascade for the eyes.
    

#for i in images:
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5) # We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
for (x, y, w, h) in faces: # For each detected face:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2) # We paint a rectangle around the face.
    roi_gray = gray[y:y+h, x:x+w] # We get the region of interest in the black and white image.
    roi_color = img[y:y+h, x:x+w] # We get the region of interest in the colored image.
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=7) # We apply the detectMultiScale method to locate one or several eyes in the image.
    for (ex, ey, ew, eh) in eyes: # For each detected eye:
        cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2) # We paint a rectangle around the eyes, but inside the referential of the face.
cv2.imshow('img', img)

cv2.waitKey(0)
cv2.destroyAllWindows() # We destroy all the windows inside which the images were displayed.