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

def invert_dataframe(original_dataframe):
    """Inverts the dataframe by simply multiplying all values by -1.

    Args:
        original_datatframe (df): The dataframe to be inverted.

    Returns:
        inverted_dataframe (df): The inverted dataframe.

    """
    inverted_dataframe = original_dataframe.multiply(-1)
    return(inverted_dataframe)

def combine_dataframe(first_dataframe, second_dataframe):
    """Combines the dataframes. Assumes that both dataframes have the same columns

    Args:
        first_datatframe (df): The first dataframe to be combined.
        second_datatframe (df): The second dataframe to be combined.

    Returns:
        combined_dataframe (df): The combined dataframe.

    """
    combined_dataframe = pd.concat([first_dataframe, second_dataframe])
    return(combined_dataframe)

def invert_and_combine(original_dataframe):
    """Inverts and combines the dataframes. Assumes that both dataframes have the same columns

    Args:
        original_dataframe (df): The dataframe to be inverted and combined with the original.

    Returns:
        new_dataframe (df): The combined dataframe.

    """
    inverted_dataframe = invert_dataframe(original_dataframe)
    new_dataframe = combine_dataframe(original_dataframe, inverted_dataframe)
    return(new_dataframe)

X = invert_and_combine(X)
y = invert_and_combine(y)


C = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
penalty = ['l2']
solver = ['liblinear']
fit_intercept=[False]
param_grid = dict(C=C, penalty=penalty, fit_intercept=fit_intercept, solver=solver)

lr = LogisticRegression()
grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv = 3, n_jobs=-1, scoring='accuracy')
best_model_from_cv = grid.fit(X, y)

print("Best model according to grid search: {0} using {1}".format(round(best_model_from_cv.best_score_,2), best_model_from_cv.best_params_))

# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
means = best_model_from_cv.cv_results_['mean_test_score']
stds = best_model_from_cv.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, best_model_from_cv.cv_results_['params']):
    print("{0:.3f} (+/-{1:.3f}) for {2}".format(mean, std * 2, params))
    
# TODO: clean this up
target_accuracy = best_model_from_cv.cv_results_['mean_test_score'][best_model_from_cv.best_index_] - best_model_from_cv.cv_results_['std_test_score'][best_model_from_cv.best_index_]
target_index = best_model_from_cv.best_index_
for i, score in enumerate(best_model_from_cv.cv_results_['mean_test_score']):
    if(score > target_accuracy and i < target_index):
        target_index = i
        print("Found better model: {0:.3f} (+/-{1:.3f}) using {2}".format(best_model_from_cv.cv_results_['mean_test_score'][target_index],2*best_model_from_cv.cv_results_['std_test_score'][target_index], best_model_from_cv.cv_results_['params'][target_index]))
        break
        
best_model_parameters = best_model_from_cv.cv_results_['params'][target_index]

# TODO: must be a cleaner way to import GridSearchCV into LogisticRegression
final_model_with_all_data = LogisticRegression(penalty=best_model_parameters['penalty'], 
                                 C=best_model_parameters['C'],
                                 fit_intercept=best_model_parameters['fit_intercept'],
                                 solver=best_model_parameters['solver'])

final_model_with_all_data = final_model_with_all_data.fit(X, y)

final_model_with_all_data = final_model_with_all_data.fit(X, y)

features = list(df.columns)
features.remove("Outcome")
[coef] = final_model_with_all_data.coef_.tolist()

rounded_coef = []
for number in coef:
    rounded_number = round(number, 2)
    rounded_coef.append(rounded_number)

x = zip(rounded_coef, features)
print(sorted(list(x)))

result = final_model_with_all_data.predict_proba([[0,0,1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

print("The probability that Marc beats Rushi is {0}%".format(round(result[0][0]*100,2)))

result = final_model_with_all_data.predict_proba([[-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

print("The probability that Shaq beats Gray is {0}%".format(round(result[0][0]*100,2)))

result = final_model_with_all_data.predict_proba([[-1,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1]])

print("The probability that Shaq and Gray beat Rushi is {0}%".format(round(result[0][0]*100,2)))

result = final_model_with_all_data.predict_proba([[0,0,0,-1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]])

print("The probability that Marc beats Sam is {0}%".format(round(result[0][0]*100,2)))

result = final_model_with_all_data.predict_proba([[0,0,1,-1,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,-1]])

print("The probability that Marc and Sam beat Rushi is {0}%".format(round(result[0][0]*100,2)))

result = final_model_with_all_data.predict_proba([[0,0,1,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0]])

print("The probability that Vic beats Rushi is {0}%".format(round(result[0][0]*100,2)))