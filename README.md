# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the required packages. 
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MANO KARTHICK S 
RegisterNumber: 212222230077
*/
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### Data head()
![image](https://github.com/MANOKARTHICK09/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121785458/245c817a-daf5-432c-bc9d-c660e4568616)


### Data info()
![image](https://github.com/MANOKARTHICK09/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121785458/58a4d80e-1112-4ce2-860a-a220a59f84b8)


#### Data.isnull().sum()
![image](https://github.com/MANOKARTHICK09/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121785458/72e57b62-23d6-49aa-8046-3df8d2a338a1)


### y_pred
![image](https://github.com/MANOKARTHICK09/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121785458/617a6ab7-aafa-42c8-bc9d-8297be304035)


### Accuracy
![image](https://github.com/MANOKARTHICK09/Implementation-of-SVM-For-Spam-Mail-Detection/assets/121785458/5c9b8216-f45c-4de7-94b1-840e448b1658)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
