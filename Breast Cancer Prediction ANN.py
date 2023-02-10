import tensorflow
import numpy as np
import pandas as pd

dataset = pd.read_csv('data (1).csv')
X = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy='mean')
imputer.fit(X[:, 2:])
imputer.transform(X[:,2:])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(X_train)
sc.transform(X_test)



from tensorflow import keras


ann = keras.models.Sequential()
ann.add(keras.layers.Dense(units = 6, activation = 'relu'))
ann.add(keras.layers.Dense(units = 6, activation = 'relu'))
ann.add(keras.layers.Dense(units = 1, activation = 'sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
print("ANN Accuracy = " + str(score))
