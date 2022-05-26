""" 
REGRESSION - Finding out relationship between two variables

Linear regression - When x & y are making a linear relationship (a straight line)
Logistic regression - when y is binary - 0,1 T,F Yes,No

Multilinear Regression - Multiple Factors that y is dependent upon
"""
import csv 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
df = pd.read_csv("Adimission_Predict.csv")
scores = df[["GRE Score","TOEFL Score"]]
results = df["Chance of admit"]

strain,stest,rtrain,rtest = train_test_split(scores,results,test_size=0.25,random_state=0)
model = LogisticRegression(random_state = 0)
model.fit(strain,rtrain)
rpredict = model.predict(stest)

from sklearn.metrics import accuracy_score

acc = accuracy_score(rtest,rpredict)
print(acc)