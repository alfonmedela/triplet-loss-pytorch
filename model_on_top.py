import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':

    root = 'embeddings/'

    x_train, y_train = np.load(root + 'x_train.npy'), np.load(root + 'y_train.npy')
    x_val, y_val = np.load(root + 'x_val.npy'), np.load(root + 'y_val.npy')

    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    score = knn.score(x_val, y_val)
    print('KNN:', score)

    SVM = SVC()
    SVM.fit(x_train, y_train)
    score = SVM.score(x_val, y_val)
    print('SVM:', score)

    L_SVM = LinearSVC()
    L_SVM.fit(x_train, y_train)
    score = L_SVM.score(x_val, y_val)
    print('Linear SVM:', score)

    RF = RandomForestClassifier()
    RF.fit(x_train, y_train)
    score = RF.score(x_val, y_val)
    print('RF:', score)