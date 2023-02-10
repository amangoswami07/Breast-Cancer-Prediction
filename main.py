import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('data (1).csv')
X = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 1].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 2:])
imputer.transform(X[:, 2:])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =0 )

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier1 = GaussianNB()
classifier1.fit(X_train, y_train)

from sklearn.neighbors import KNeighborsClassifier
classifier2 = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p=2)
classifier2.fit(X_train, y_train)

from sklearn.linear_model import LogisticRegression
classifier3 = LogisticRegression(random_state=0)
classifier3.fit(X_train, y_train)

from sklearn.svm import SVC
classifier4 = SVC(kernel = 'rbf', random_state = 0)
classifier4.fit(X_train, y_train)


y_pred1 = classifier1.predict(X_test)
y_pred2 = classifier2.predict(X_test)
y_pred3 = classifier3.predict(X_test)
y_pred4 = classifier4.predict(X_test)



from sklearn.metrics import accuracy_score
score1 = accuracy_score(y_test, y_pred1)
score2 = accuracy_score(y_test, y_pred2)
score3 = accuracy_score(y_test, y_pred3)
score4 = accuracy_score(y_test, y_pred4)
print("Naive-Bayes Accuracy = " + str(score1))
print("K-Nearest Neighbors Accuracy = " + str(score2))
print("Logistic Regression Accuracy = " + str(score3))
print("Kernel SVM Accuracy = " + str(score4))
