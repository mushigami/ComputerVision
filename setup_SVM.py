import cv2
import numpy as np

def svmInit(C, gamma):
  model = cv2.ml.SVM_create()
  model.setGamma(gamma)
  model.setC(C)
  model.setKernel(cv2.ml.SVM_RBF)
  model.setType(cv2.ml.SVM_C_SVC)
  # model.setDegree(4)

  return model

def svmTrain(model, samples, responses):
  model.train(samples, cv2.ml.ROW_SAMPLE, responses)
  return model

def svmPredict(model, samples):
  return model.predict(samples)[1].ravel()

def svmEvaluate(model, samples, labels):
  predictions = svmPredict(model, samples)
  accuracy = (labels == predictions).mean()
  print('Percentage Accuracy: %.2f %%' % (accuracy * 100))
  return accuracy

def prepareData(data):
  featureVectorLength = len(data[0])
  features = np.float32(data).reshape(-1, featureVectorLength)
  return features