##Dataset Link: https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data
import pickle
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pathlib
from datetime import datetime

PATH = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(PATH, "BankNote_Authentication.csv" )
pathlib.Path(os.path.join(PATH, "models")).mkdir(exist_ok=True) 

df=pd.read_csv(data_path)
x, y = df.iloc[:,:-1], df.iloc[:,-1]
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3)

print(x_train.head(5))

classifier=RandomForestClassifier()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

score=accuracy_score(y_test,y_pred)

print(score)

timestamp = "{:%Y%m%d_%H%M%S}".format(datetime.now())
print(timestamp)
model_path = os.path.join(PATH, f"models/{timestamp}.pkl")

with open(model_path, 'wb') as file:  
    pickle.dump(classifier, file)

file.close()

# classifier.predict([[2,3,4,1]])





