import cv2
import numpy as np
import cropFace
from dataPath import DATA_PATH
import os

predictions2Label = {0: "No Face", 1: "Face"}

def getImages(path, class_val, test_fraction = 0.2):
  testData = []
  trainData = []
  trainLabels = []
  testLabels = []
  inputDir = os.path.expanduser(path)

  # Get images from the directory and find number of train
  # and test samples
  if os.path.isdir(inputDir):
    images = os.listdir(inputDir)
    images.sort()
    nTest = int(len(images) * test_fraction)

  for counter, img in enumerate(images):

    im = cv2.imread(os.path.join(inputDir, img))
    # Add nTest samples to testing data
    if counter < nTest:
      testData.append(im)
      testLabels.append(class_val)
    else:
      # Add nTrain samples to training data
      trainData.append(im)
      trainLabels.append(class_val)

  return trainData, trainLabels, testData, testLabels