"""Transformations Utility Functions

This module stores transformation functions that multiple models use.
This is broken out for usability purposes.

Todo:

"""

import pandas as pd


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