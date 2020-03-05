import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('./data/models/haarcascade_frontalface_default.xml')

def getCroppedEyeRegion(targetImage):
    
    targetImageGray = cv2.cvtColor(targetImage, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(targetImageGray,1.3,5)
    x,y,w,h = faces[0]

    face_roi = targetImage[y:y+h,x:x+w]
    cropped = cv2.resize(face_roi,(96, 96), interpolation = cv2.INTER_CUBIC)
    cv2.imshow('test',cropped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return cropped
