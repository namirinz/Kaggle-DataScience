import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

df = pd.read_csv('datasets/diabetes.csv')
df = df[(df.Glucose != 0)&(df.BloodPressure != 0)&(df.BMI != 0)]
label = df.Outcome
df = df.drop('Outcome',axis = 1)

X = df.values
y = label.values

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)

steps2 = [('scaler',StandardScaler()),('logreg',LogisticRegression())]
param_grid = {'logreg__penalty': ['l1','l2'], 'logreg__C': np.linspace(0.001,1)}

pipeline2 = Pipeline(steps2)

pipeline_cv2 = GridSearchCV(pipeline2,param_grid)

pipeline2.fit(X_train,y_train)

y_pred2 = pipeline2.predict(X_test)

print("Accuracy :",accuracy_score(y_test,y_pred2))
print(classification_report(y_test,y_pred2))



