"""Ranking

This file is used to streamline script execution from the command line.

"""

# TODO: Clean this up
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np

df = pd.read_csv("Data/sample_data.csv")

X = df.loc[:, df.columns != 'Outcome']
y = df.Outcome


def invert_dataframe(original_dataframe):
    """Inverts the dataframe by simply multiplying all values by -1.

    Args:
        original_dataframe (df): The dataframe to be inverted.

    Returns:
        inverted_dataframe (df): The inverted dataframe.

    """
    inverted_dataframe = original_dataframe.multiply(-1)
    return inverted_dataframe


def combine_dataframe(first_dataframe, second_dataframe):
    """Combines the dataframes. Assumes that both dataframes have the same columns

    Args:
        first_dataframe (df): The first dataframe to be combined.
        second_dataframe (df): The second dataframe to be combined.

    Returns:
        combined_dataframe (df): The combined dataframe.

    """
    combined_dataframe = pd.concat([first_dataframe, second_dataframe])
    return combined_dataframe


def invert_and_combine(original_dataframe):
    """Inverts and combines the dataframes. Assumes that both dataframes have the same columns

    Args:
        original_dataframe (df): The dataframe to be inverted and combined with the original.

    Returns:
        new_dataframe (df): The combined dataframe.

    """
    inverted_dataframe = invert_dataframe(original_dataframe)
    new_dataframe = combine_dataframe(original_dataframe, inverted_dataframe)
    return new_dataframe


X = invert_and_combine(X)
y = invert_and_combine(y)


C = np.logspace(-1,4,1000)
penalty = ['l2']
solver = ['liblinear']
fit_intercept = [False]
param_grid = dict(C=C, penalty=penalty, fit_intercept=fit_intercept, solver=solver)

lr = LogisticRegression()
grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
cross_validation_models = grid.fit(X, y)

print("Best model according to grid search: {0} using {1}".format(
    round(cross_validation_models.best_score_, 2), cross_validation_models.best_params_))


def find_target_accuracy(cv_models):
    """Finds the target accuracy that the second best model has to exceed

    Args:
        cv_models (GridSearchCV): The object that has the info from cross validation.

    Returns:
        target_accuracy (float): The best model's accuracy descreased by its standard deviation

    """
    best_cv_model_index = cv_models.best_index_
    best_cv_model_mean_accuracy = cv_models.cv_results_['mean_test_score'][best_cv_model_index]
    best_cv_model_std_accuracy = cv_models.cv_results_['std_test_score'][best_cv_model_index]
    target_accuracy = best_cv_model_mean_accuracy - best_cv_model_std_accuracy
    return target_accuracy


def find_final_model_params(cv_models):
    """Finds the parameters for the final model that will be trained on all data.
    We want to see whether there is a model that has more generalization but satisfactory accuracy

    Args:
        cv_models (GridSearchCV): The object that has the info from cross validation.

    Returns:
        final_model_params (dict): The final model's parameters

    """
    target_accuracy = find_target_accuracy(cv_models)
    index_of_final_model = loop_through_cv_to_find_index_of_final_model(cv_models, target_accuracy)
    final_model_params = cv_models.cv_results_['params'][index_of_final_model]
    return final_model_params


def loop_through_cv_to_find_index_of_final_model(cv_models, target_accuracy):
    """We want to see whether there is a model that has more generalization but
    satisfactory accuracy. We loop through the results until we find a model
    that has a satisfactory accuracy. The loop will stop at the best model in
    case none of the models with higher generalization are satisfactory.
    This function is assuming that the models are sorted.

    Args:
        cv_models (GridSearchCV): The object that has the info from cross validation.
        target_accuracy (float): The cv best model's accuracy descreased by its standard deviation

    Returns:
        target_index (int): The index of the final model

    """
    target_index = cv_models.best_index_
    for i, score in enumerate(cv_models.cv_results_['mean_test_score']):
        if score > target_accuracy and i < target_index:
            target_index = i
            print("Found adequate model with better generalization: {0:.3f} (+/-{1:.3f}) using {2}".
                  format(cv_models.cv_results_['mean_test_score'][target_index],
                         2 * cv_models.cv_results_['std_test_score'][target_index],
                         cv_models.cv_results_['params'][target_index]))
            break
    return target_index


best_model_parameters = find_final_model_params(cross_validation_models)

# TODO: must be a cleaner way to import GridSearchCV into LogisticRegression
final_model_with_all_data = LogisticRegression(penalty=best_model_parameters['penalty'],
                                               C=best_model_parameters['C'],
                                               fit_intercept=best_model_parameters['fit_intercept'],
                                               solver=best_model_parameters['solver'])

final_model_with_all_data = final_model_with_all_data.fit(X, y)

features = list(df.columns)
features.remove("Outcome")
[coef] = final_model_with_all_data.coef_.tolist()

rounded_coef = []
for number in coef:
    rounded_number = round(number, 2)
    rounded_coef.append(rounded_number)

x = zip(rounded_coef, features)
print("Final model coefficients are: {0}".format(sorted(list(x))))

result = final_model_with_all_data.predict_proba(
    [[0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
print("The probability that Marc beats Rushi is {0}%".format(round(result[0][0]*100, 2)))

result = final_model_with_all_data.predict_proba(
    [[-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
print("The probability that Shaq beats Gray is {0}%".format(round(result[0][0]*100, 2)))

result = final_model_with_all_data.predict_proba(
    [[-1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]])
print("The probability that Shaq and Gray beat Rushi is {0}%".format(round(result[0][0]*100, 2)))

result = final_model_with_all_data.predict_proba(
    [[0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
print("The probability that Marc beats Sam is {0}%".format(round(result[0][0]*100, 2)))

result = final_model_with_all_data.predict_proba(
    [[0, 0, 1, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]])
print("The probability that Marc and Sam beat Rushi is {0}%".format(round(result[0][0]*100, 2)))

result = final_model_with_all_data.predict_proba(
    [[0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
print("The probability that Vic beats Rushi is {0}%".format(round(result[0][0]*100, 2)))
