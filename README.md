# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.


## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: 212222240118
RegisterNumber:  YOHESH KUMAR R.M
```
```
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
### Result Output:
![ro](https://github.com/yoheshkumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393568/8153776e-f7d8-4720-b557-20dedbefa648)
### data.head( ):
![dhh](https://github.com/yoheshkumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393568/da9c6031-f629-4010-8fd2-2f96fcb6e93c)
### data.info( ):
![dii](https://github.com/yoheshkumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393568/8ef1631c-6bf3-4671-9471-d7be75bac7bf)
### data.isnull().sum():
![dii ss](https://github.com/yoheshkumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393568/b7da571c-47a9-4b7c-b441-e00673e41599)
### Y_prediction:
![ypredd](https://github.com/yoheshkumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393568/2d2aeb4f-910a-47cc-8c53-e219437a31b2)
### Accuracy Value:
![av](https://github.com/yoheshkumar/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393568/4f5d32c3-f364-4163-85eb-f905e40010b9)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
