"""Predictions Utility Functions

This module stores prediction functions that multiple models use.
This is broken out for usability purposes.

Todo:

- Fix the hardcoding of people

"""

def make_predictions(model):
    """
    We want to use this model to determine the probability of a game. Each value in the array 
    corresponds to a person. For example, the first number is Shaq, the second number is Gray, etc.
    
    Args:
        model(sklearn model): Classifier model that we are using to predict probalities
        
    Return:
        None
    """
    result = model.predict_proba([[0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    print("The probability that Marc beats Rushi is {0}%".format(round(result[0][0]*100, 2)))
    
    result = model.predict_proba([[-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    print("The probability that Shaq beats Gray is {0}%".format(round(result[0][0]*100, 2)))

    result = model.predict_proba([[-1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]])
    print("The probability that Shaq and Gray beat Rushi is {0}%".format(round(result[0][0]*100, 2)))

    result = model.predict_proba([[0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    print("The probability that Marc beats Sam is {0}%".format(round(result[0][0]*100, 2)))

    result = model.predict_proba([[0, 0, 1, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]])
    print("The probability that Marc and Sam beat Rushi is {0}%".format(round(result[0][0]*100, 2)))

    result = model.predict_proba([[0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    print("The probability that Vic beats Rushi is {0}%".format(round(result[0][0]*100, 2)))
