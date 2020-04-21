import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn


df = pd.read_csv("sample_data.csv")
X = df.loc[:, df.columns != 'Outcome'] # Features
y = df.Outcome # Target variable

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

# instantiate the model (using the default parameters)
logreg = LogisticRegression(fit_intercept=False)

# fit the model with data
logreg.fit(X_train,y_train)

features = list(df.columns)
features.remove("Outcome")
[coef] = logreg.coef_.tolist()

rounded_coef = []
for number in coef:
    rounded_number = round(number, 2)
    rounded_coef.append(rounded_number)

x = zip(rounded_coef, features)
print(sorted(list(x)))

# predict
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
