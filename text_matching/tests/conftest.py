"""
Pytest configuration file for the text_prediction test suite.

This module provides shared fixtures and configuration for testing
the text_prediction package.
"""

import numpy as np
import pytest

from typing import List, Dict

########################################################################
## SHARED TEST DATA
########################################################################

@pytest.fixture
def string_choices() -> List[str]:
    """
    Fixture providing a list of string choices for predictor testing.
    
    Returns
    -------
    List[str]
        A list of sample strings.
    """

    # Create a list of choices.
    choices = ["apple", "banana", "orange", "pineapple", "strawberry"]

    return choices

@pytest.fixture
def document_data() -> List[str]:
    """
    Fixture providing a list of document strings for fitting.
    
    Returns
    -------
    List[str]
        A list of sample documents.
    """

    # Create a list of documents.
    documents = [
        "I like apples", "Bananas are yellow", "Oranges are citrus fruits",
        "Pineapple is tropical", "Strawberries are red"
    ]

    return documents

@pytest.fixture
def query_data() -> List[str]:
    """
    Fixture providing a list of query strings.
    
    Returns
    -------
    List[str]
        A list of sample queries.
    """

    # Create a list of queries.
    queries = ["aple", "banana", "orang", "pineaple", "strawbery"]

    return queries

@pytest.fixture
def mock_vector_data() -> Dict[str, np.ndarray]:
    """
    Fixture providing mock vector data for testing.
    
    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary with keys 'X' and 'y' containing mock feature
        and target data.
    """

    # Create a simple 2D feature matrix.
    X = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0]
    ])
    
    # Create a target vector.
    y = np.array([0, 1, 2, 3, 4])
    
    # Return the feature matrix and target vector.
    vector_data = {"X": X, "y": y}

    return vector_data