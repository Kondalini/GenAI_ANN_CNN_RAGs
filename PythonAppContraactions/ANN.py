from keras.src.optimizers import optimizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense






dataset = pd.read_csv(r'C:\Users\akrastev\source\repos\PythonAppPandas\PythonAppPandas\Churn_Modelling.csv')

X=dataset.iloc[:,3:13]
y=dataset.iloc[:,13]  
print(dataset.head(5))
print(X)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lableencoder_X_1 = LabelEncoder()
X[:,1] = lableencoder_X_1.fit_transform(X[:,1])
lableencoder_X_2 = LabelEncoder()
X[:,2] = lableencoder_X_2.fit_transform(X[:,2])

#Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=0)
print(len(dataset))
print(len(X_test))

#Feature scaling 
from sklearn.preprocesing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Clasical Ml
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
pred =clf.predict(X_test)

#making a confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,pred)
print(cm)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,pred)
print(score*100)

#Accurasy from Desision Tree Clasifier :80%

#random forest Clasical ML
from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier(n_estimators=200)
clf2.fit(X_train,y_train)
pred2 =clf2.predict(X_test)

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test,pred)
print(cm2)
from sklearn.metrics import accuracy_score
score2 = accuracy_score(y_test,pred2)
print(score2*100)

#Accurasy from Desision Tree Clasifier :86%

#ANN Model
#Initiliasing the Ann
classifier = Sequential()
#units:  it is an art, comes by experience,input+output/2
#niliciazer: how your weights are updated
#relu activation function
#input_dim: imput shape (shape of the date, how many features do we have in the data
#Adding the input layer and the first hidden layer 
classifier.add(Dense(units=6,kernel_initializer = 'uniform', activation = 'relu',input_dim =10))
#Adding second hiden layer 
classifier.add(Dense(units=6,kernel_initializer = 'uniform', activation = 'relu'))
#Adding third hiden layer 
classifier.add(Dense(units=6,kernel_initializer = 'uniform', activation = 'relu'))

#Adding the output layer
classifier.add(Dense(units=1,kernel_initializer = 'uniform', activation = 'sigmoid'))

print(len(X_train))

#Compiling the ANN
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics =['accuracy'])
y_train[0]
#Fitting the ANN to the Training set
classifier.fit(X_train,y_train,batch_size = 10,epochs =200)

#part3 Making predictions and evaluating a model
y_pred_ann = classifier.predict(X_test)
y_pred_ann = (y_pred_ann>0,5)
cm3 = confusion_matrix(y_test,y_pred_ann)
print(cm3) #

#making the confusion matrix
from sklearn.metrics import accuracy_score
score3 = accuracy_score(y_test, y_pred_ann)

#Accuracy 84%



