"""
Additional helper scorers for common metrics.
"""

########################################################################
## IMPORTS
########################################################################

from typing import List

########################################################################
## SCORERS
########################################################################

def total_queries_scorer(y_true: List[str], y_pred: List[str]) -> int:
    """
    Return the total number of queries.

    Parameters
    ----------
    y_true : array-like
        The true labels.
    y_pred : array-like
        The predicted labels.

    Returns
    -------
    int
        The total number of queries.
    """
    return len(y_true)

def correct_matches_scorer(y_true: List[str], y_pred: List[str]) -> int:
    """
    Return the number of correct matches.

    Parameters
    ----------
    y_true : array-like
        The true labels.
    y_pred : array-like
        The predicted labels.

    Returns
    -------
    int
        The number of correct matches.
    """
    return sum(1 for t, p in zip(y_true, y_pred) if t == p)

def incorrect_matches_scorer(y_true: List[str], y_pred: List[str]) -> int:
    """
    Return the number of incorrect matches.

    Parameters
    ----------
    y_true : array-like
        The true labels.
    y_pred : array-like
        The predicted labels.

    Returns
    -------
    int
        The number of incorrect matches.
    """
    return sum(1 for t, p in zip(y_true, y_pred) if t != p)
