# TODO: Clean this up
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

df = pd.read_csv("sample_data.csv")

# Designate all columns that are not `Outcome` as features and `Outcome` as target
X = df.loc[:, df.columns != 'Outcome']
y = df.Outcome

# X_train,X_validate,y_train,y_validate=train_test_split(X,y,test_size=0.20,random_state=0)

max_iter=[100,110,120,130,140]
C = [0.01, 0.1, 1, 10, 100, 1000]
penalty = ['l1','l2']
solver = ['newton-cg', 'lbfgs', 'sag', 'saga', 'liblinear']
fit_intercept=[False]
param_grid = dict(max_iter=max_iter,C=C, penalty=penalty, fit_intercept=fit_intercept, solver=solver)

lr = LogisticRegression()
grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv = 3, n_jobs=-1, scoring='accuracy')
best_model = grid.fit(X, y)

print("Best score: {0} using {1}".format(round(best_model.best_score_,2), best_model.best_params_))

# y_true, y_pred = y_validate, best_model.predict(X_validate)
# print(classification_report(y_true, y_pred))

final_model = LogisticRegression(penalty=best_model.best_params_['penalty'],
                                 C=best_model.best_params_['C'],
                                 fit_intercept=best_model.best_params_['fit_intercept'],
                                 max_iter=best_model.best_params_['max_iter'],
                                 solver=best_model.best_params_['solver'])

final_model = final_model.fit(X, y)

features = list(df.columns)
features.remove("Outcome")
[coef] = final_model.coef_.tolist()

rounded_coef = []
for number in coef:
    rounded_number = round(number, 2)
    rounded_coef.append(rounded_number)

x = zip(rounded_coef, features)
print(sorted(list(x)))

result = final_model.predict_proba([[0,0,1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

print("The probability that Marc beats Rushi is {0}%".format(round(result[0][0]*100,2)))

result = final_model.predict_proba([[-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

print("The probability that Shaq beats Gray is {0}%".format(round(result[0][0]*100,2)))

result = final_model.predict_proba([[-1,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1]])

print("The probability that Shaq and Gray beat Rushi is {0}%".format(round(result[0][0]*100,2)))
result = final_model.predict_proba([[0,0,0,-1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]])

print("The probability that Marc beats Sam is {0}%".format(round(result[0][0]*100,2)))

result = final_model.predict_proba([[0,0,1,-1,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,-1]])

print("The probability that Marc and Sam beat Rushi is {0}%".format(round(result[0][0]*100,2)))

result = final_model.predict_proba([[0,0,1,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0]])

print("The probability that Vic beats Rushi is {0}%".format(round(result[0][0]*100,2)))