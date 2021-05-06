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

def getData():
    class0Directory = r"C:\Users\hoefs\Desktop\Altered_Celegens_ModelGen\\0"
    class1Directory = r"C:\Users\hoefs\Desktop\Altered_Celegens_ModelGen\\1"

    dataPath0 = os.path.join(class0Directory,'*g')
    c0Files = glob.glob(dataPath0)
    dataPath1 = os.path.join(class1Directory,'*g')
    c1Files = glob.glob(dataPath1)

    #X , y
    X = []
    y = []
    # import c0 files
    for file in c0Files:
        image = cv2.imread(file)
        # feature = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # feature = np.asarray(feature)
        # X.append(np.array(feature))
        feature = im_process(image,False)
        X.append(np.array(feature))
        y.append(0)
    
    for file in c1Files:
        image = cv2.imread(file)
        # feature = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # feature = np.asarray(feature)
        feature = im_process(image,False)
        X.append(np.array(feature))
        y.append(1)

    return X,y

def imadjust(image, inlo, inhi, outlo, outhi, gamma):
    n_image = image/np.max(image)
    return 255*((outhi-outlo)*(n_image-inlo)/(inhi-inlo)**gamma + outlo)


def im_process(image, do_gradient=True):
    g_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    g_transform = imadjust(g_image, 0, 1, 0, 1, 3)
    
    if do_gradient:
        smoothed = cv2.GaussianBlur(g_transform, (5, 5), 2)
        gradient = cv2.Laplacian(smoothed, cv2.CV_64F)
        equalized = np.absolute(gradient)
        normalized = equalized/np.max(equalized)
        return np.uint8(255*normalized)
    else:
        return np.uint8(g_transform)

    
def performPCA(x_train,x_test):
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    startTime = time.time()
    pca = PCA(0.95)
    pca.fit(x_train)
    joblib.dump(pca,"pcaFit.pkl")

    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    print("--- %s minutes for PCA---" % str(((time.time() - startTime)/60)))
    return x_train,x_test




X,y = getData()
X_train,X_test,y_train,y_test = train_test_split(X,y,shuffle=True,train_size=.95)

# Reformat Data     
x__train = np.zeros([len(X_train), 101,101])
x__test = np.zeros([len(X_test), 101,101])

for i in range(len(X_train)):
    x__train[i] = X_train[i]
    
for i in range(len(X_test)):
    x__test[i] = X_test[i]

X_train = x__train
X_test = x__test

mean = 0
std = 0
x_train = X_train.reshape(X_train.shape[0], 101*101)
mean = np.mean(x_train)
std = np.std(x_train)
x_train = (x_train-mean)/std
new_col = np.ones(len(x_train)).reshape(len(x_train),1)
x_train =  np.append(x_train,new_col,axis=1)
x_test = X_test.reshape(X_test.shape[0], 101*101)
x_test = (x_test-mean)/std
new_col = np.ones(len(x_test)).reshape(len(x_test),1)
x_test =  np.append(x_test,new_col,axis=1)  
y_train = np.array(y_train,dtype= 'f')
y_test = np.array(y_test,dtype= 'f')

# Perform PCA
x_train,x_test = performPCA(x_train,x_test)

clf = svm.SVC(C = 20 ,kernel = 'rbf')
startTime = time.time()
clf.fit(x_train,y_train)
print("--- %s seconds for training---" % (time.time() - startTime))

start_time = time.time()

y_pred = clf.predict(x_test)
print("--- %s seconds for testing---" % (time.time() - start_time))
print("SCORE:")
print(clf.score(x_test,y_test))
print("CONFUSION MATRIX:")
print(confusion_matrix(y_test,y_pred))
print("CLASSIFICATION REPORT")
print(classification_report(y_test,y_pred))
print("support vectors:")
print(clf.support_vectors_)
print("Coefficients:")
print(clf.dual_coef_)


joblib.dump(clf,'TrainedModel.pkl',compress=9)