import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split

raw_data = pd.read_csv("./data/train.csv").values
X_train, X_test, y_train, y_test = \
    train_test_split(raw_data[:,:-1], raw_data[:,-1],
                     test_size=0.1, random_state = 1234)

# linear svm classfier
svc = SVC(C=1, kernel='linear')
svc.fit(X_train, y_train)


