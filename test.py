
import numpy as np
import cv2
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn import svm
import time
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

# load in data from file path (This will be the path sarraf provides)
def loadData():
    directory = r"C:\Users\hoefs\Documents\Celegans_ModelGen\\test"
    dataPath = os.path.join(directory,'*g')
    testFiles = glob.glob(dataPath)
    return testFiles

testFiles = loadData()  
# load in the trained model from SVM and PCA
trainedModel = joblib.load('TrainedModel.pkl')
pcaFit = joblib.load('pcaFit.pkl')

# loop through the files in the provided path and predict 
for file in testFiles:
    X = []
    image = cv2.imread(file)
    feature = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = np.asarray(feature)
    X.append(feature)
   
    x_test = np.zeros([len(X),101,101])
    for i in range(len(X)):
        x_test[i] = X[i]
    
    # Reformat data
    mean = np.mean(x_test)
    std = np.std(x_test)
    X_test = x_test.reshape(x_test.shape[0], 101*101)
    X_test = (X_test-mean)/std
    new_col = np.ones(len(X_test)).reshape(len(X_test),1)
    X_test =  np.append(X_test,new_col,axis=1)
    #Perform PCA 
    X_test = pcaFit.transform(X_test)
    # Predict
    y_pred = trainedModel.predict(X_test)
    # Print filename and prediction to console 
    print('File: ', file, 'Classification: ', y_pred)


