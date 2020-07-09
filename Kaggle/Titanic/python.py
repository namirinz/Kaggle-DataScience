import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

pd.set_option("display.max_columns",500)

df_train = pd.read_csv("datasets/train.csv")
df_test = pd.read_csv("datasets/test.csv")
df_ans = pd.read_csv("datasets/gender_submission.csv")

#print(df_train.info())
#print(df_train.describe())

col_drop = ['Cabin','PassengerId','Name','Ticket']
col_dummies = ['Sex','Embarked']

df_train.drop(columns=col_drop,inplace = True)
df_train = df_train[df_train['Embarked'].notna()]
X = df_train.drop('Survived',axis = 1)
y = df_train['Survived']

X = X.fillna(np.nan)
X = pd.get_dummies(X,drop_first = True)

df_test.drop(columns=col_drop, inplace = True)
X_test = df_test[df_test['Embarked'].notna()]

X['Age'] = X.Age.apply(lambda x: X.Age.mean() if np.isnan(x) else x)
X_test['Age'] = X_test.Age.apply(lambda x: X_test.Age.mean() if np.isnan(x) else x)
X_test['Fare'] = X_test.Fare.apply(lambda x: X_test.Fare.mean() if np.isnan(x) else x)
X_test = pd.get_dummies(X_test, drop_first = True)
y_test = df_ans.drop('PassengerId', axis = 1)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

knn = KNeighborsClassifier()
logreg = LogisticRegression(solver='liblinear')
svc = SVC()
dt = RandomForestClassifier(min_samples_leaf = 0.1)

cat_col = ['Pclass','Sex_male','Embarked_Q','Embarked_S']
num_col = [i for i in X.columns if i not in cat_col]

#print(cat_col)
#print(num_col)
feature_union = FeatureUnion([
      ('category', FunctionTransformer(lambda x: x[cat_col])),
      ('numeric', Pipeline([
        ('select', FunctionTransformer(lambda x: x[num_col])),
        ('scale', StandardScaler()),
        ('PCA', PCA())  
        ])
      )   
])

pca_grid = '__feature_select__numeric__PCA__n_components'
pca_grid_value = [2]

knn_union = Pipeline([('feature_select',feature_union),('KNN',knn)])
logreg_union = Pipeline([('feature_select',feature_union), ('LOGREG', logreg)])
svc_union = Pipeline([('feature_select',feature_union), ('SVC',svc)])
vote = VotingClassifier(estimators=[('knn',knn_union),('log',logreg_union),('svc',svc_union), ('dt', dt)])
params = {'knn__KNN__n_neighbors': [2,3,4,5,6], 'log__LOGREG__penalty': ['l1','l2'], 'log__LOGREG__C': [0.001, 0.01, 0.1, 1],'svc__SVC__C': [0.001, 0.01, 0.1, 1], 'knn'+pca_grid: pca_grid_value, 'log'+pca_grid: pca_grid_value, 'svc'+pca_grid: pca_grid_value, 'dt__max_features': ['sqrt','log2'], 'dt__n_estimators': [50,100,150,200], 'dt__criterion': ['gini','entropy']}

model_vote = GridSearchCV(vote, params, n_jobs = -1)
model_vote.fit(X, y)

y_train_pred = model_vote.predict(X)
y_test_pred = model_vote.predict(X_test)

print("Parameter : ",model_vote.best_params_)
print('\nTraining Score: {}, Test Score: {}'.format(model_vote.score(X, y), model_vote.score(X_test, y_test)))
print('\nClassification_report:\n',classification_report(y_test, y_test_pred))
print('\nConfusion_matrix:\n',confusion_matrix(y_test, y_test_pred))
#print()


