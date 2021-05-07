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
#Need to pip install sklearn & have cv2 installed(pip install opencv-python)


# load in data from file path (This will be the path sarraf provides)
def loadData(fileLocation):
    try:
        directory = fileLocation
        dataPath = os.path.join(directory, '*g')
        testFiles = glob.glob(dataPath)
        return testFiles
    except:
        print("Cannot Open File")
        return False


def imadjust(image, inlo, inhi, outlo, outhi, gamma):
    n_image = image / np.max(image)
    return 255 * ((outhi - outlo) * (n_image - inlo) / (inhi - inlo) ** gamma + outlo)


def im_process(image, do_gradient=True):
    g_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    g_transform = imadjust(g_image, 0, 1, 0, 1, 3)

    if do_gradient:
        smoothed = cv2.GaussianBlur(g_transform, (5, 5), 2)
        gradient = cv2.Laplacian(smoothed, cv2.CV_64F)
        equalized = np.absolute(gradient)
        normalized = equalized / np.max(equalized)
        return np.uint8(255 * normalized)
    else:
        return np.uint8(g_transform)


#Script starts Here
testFiles = False
worm = 0
noworm = 0
with open("printout.txt", "w") as f:
    while testFiles == False:
        print("Enter 'E' to exit.")
        print("Please Enter the File Location of Photos to test: ")
        fileLocation = str(input())
        if fileLocation == 'E':
            testFiles = True
        else:
            testFiles = loadData(fileLocation)

    if testFiles != True:
        # load in the trained model from SVM and PCA
        trainedModel = joblib.load('TrainedModel.pkl')
        pcaFit = joblib.load('pcaFit.pkl')

        # loop through the files in the provided path and predict
        for file in testFiles:
            X = []
            image = cv2.imread(file)
            feature = im_process(image, False)
            X.append(np.array(feature))

            x_test = np.zeros([len(X), 101, 101])
            for i in range(len(X)):
                x_test[i] = X[i]

            # Reformat data
            mean = np.mean(x_test)
            std = np.std(x_test)
            X_test = x_test.reshape(x_test.shape[0], 101 * 101)
            X_test = (X_test - mean) / std
            new_col = np.ones(len(X_test)).reshape(len(X_test), 1)
            X_test = np.append(X_test, new_col, axis=1)
            # Perform PCA
            X_test = pcaFit.transform(X_test)
            # Predict
            y_pred = trainedModel.predict(X_test)
            if int(y_pred) == 0:
                noworm = noworm + 1
            else:
                worm = worm + 1
            # Print filename and prediction to console
            printFile = file.replace(fileLocation,'')
            printFile = f"{printFile:<25}"
            print('File: ', printFile, 'Classification: ', int(y_pred))
            line = 'File: ' + printFile + 'Classification: ' + str(int(y_pred))
            f.write(line)
            f.write("\n")
    print('There are ' + str(worm) + ' worms and ' + str(noworm)  + ' pictures without worms.')
    line = 'There are ' + str(worm) + ' worms and ' + str(noworm)  + ' pictures without worms.'
    f.write(line)
    f.write("\n")
f.close()