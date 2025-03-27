"""
Tests for the FuzzyPredictor class in
text_prediction.predictors.distance.fuzzy.

This module tests the FuzzyPredictor class, which implements fuzzy
string matching using the RapidFuzz library.
"""

import pytest

from typing import List

from rapidfuzz.utils import default_process
from text_matching.predictors.distance.fuzzy import FuzzyPredictor
from text_matching.predictors.distance import ALGORITHMS, METHODS

########################################################################
## INITIALIZATION TESTS
########################################################################

def test_fuzzy_predictor_init_default() -> None:
    """
    Test initialization of FuzzyPredictor with default parameters.
    
    This test verifies that a FuzzyPredictor initialized with no
    arguments has all default values correctly set.
    """

    # Create a predictor with default parameters.
    predictor = FuzzyPredictor()
    
    # Check that all default parameters are correctly set.
    assert (
        predictor.processor == default_process
    ), "Default processor should be default_process."

    assert (
        predictor.limit == 5
    ), "Default limit should be 5."

    assert (
        predictor.score_cutoff is None
    ), "Default score_cutoff should be None."

    assert (
        predictor.score_hint is None
    ), "Default score_hint should be None."

    assert (
        isinstance(predictor.scorer_kwargs, dict)
    ), "scorer_kwargs should be a dictionary."

    assert (
        predictor.score_multiplier == 1
    ), "Default score_multiplier should be 1."

    assert (
        predictor.dtype is None
    ), "Default dtype should be None."
    assert predictor.workers == 1, "Default workers should be 1."
    
    # Check internal state for a newly created predictor.
    assert (
        predictor.raw_documents is None
    ), "raw_documents should be None for a new predictor."

    assert (
        predictor._is_fitted is False
    ), "Predictor should not be fitted by default."
    
    # Check that the scorer is properly initialized and callable.
    assert (
        callable(predictor.scorer)
    ), "Scorer should be a callable function."

def test_fuzzy_predictor_init_custom() -> None:
    """
    Test initialization of FuzzyPredictor with custom parameters.
    
    This test verifies that a FuzzyPredictor initialized with custom
    arguments has all values correctly set according to the provided
    parameters.
    """

    # Create a predictor with custom parameters.
    predictor = FuzzyPredictor(
        algorithm="jaro",
        method="similarity",
        limit=10,
        score_cutoff=0.8,
        score_hint=0.9,
        score_multiplier=2.0,
        workers=2
    )
    
    # Check that all custom parameters are correctly set.
    assert (
        predictor.limit == 10
    ), "Custom limit should be 10."

    assert (
        predictor.score_cutoff == 0.8
    ), "Custom score_cutoff should be 0.8."

    assert (
        predictor.score_hint == 0.9
    ), "Custom score_hint should be 0.9."

    assert (
        predictor.score_multiplier == 2.0
    ), "Custom score_multiplier should be 2.0."

    assert (
        predictor.workers == 2
    ), "Custom workers should be 2."
    
    # Check that the scorer is properly initialized and callable.
    assert (
        callable(predictor.scorer)
    ), "Scorer should be a callable function."

def test_fuzzy_predictor_invalid_algorithm() -> None:
    """
    Test initialization with invalid algorithm.
    
    This test verifies that initializing a FuzzyPredictor with an
    invalid algorithm name raises a ValueError.
    """

    # Attempt to create a predictor with an invalid algorithm should
    # raise ValueError.
    with pytest.raises(ValueError):
        FuzzyPredictor(algorithm="invalid_algorithm")

def test_fuzzy_predictor_invalid_method() -> None:
    """
    Test initialization with invalid method.
    
    This test verifies that initializing a FuzzyPredictor with an
    invalid method name raises a ValueError.
    """

    # Attempt to create a predictor with an invalid method should
    # raise ValueError.
    with pytest.raises(ValueError):
        FuzzyPredictor(method="invalid_method")

def test_fuzzy_predictor_all_valid_algorithms_and_methods() -> None:
    """
    Test initialization with all valid algorithms and methods.
    
    This test verifies that a FuzzyPredictor can be initialized with all
    combinations of valid algorithms and methods without errors.
    """

    # Test all valid combinations of algorithms and methods.
    for algorithm in ALGORITHMS.keys():

        # Test all valid methods for the current algorithm.
        for method in METHODS:

            # Create a predictor with the current algorithm and method.
            predictor = FuzzyPredictor(algorithm=algorithm, method=method)
            
            # Check that the scorer is properly initialized and
            # callable.
            assert (
                callable(predictor.scorer)
            ), f"Scorer should be callable for {algorithm} and {method}."

########################################################################
## FIT, TRANSFORM, FIT_TRANSFORM TESTS
########################################################################

def test_fuzzy_predictor_fit(string_choices: List[str]) -> None:
    """
    Test fit method of FuzzyPredictor.
    
    This test verifies that the fit method correctly stores the training
    data and updates the fitted state.
    
    Parameters
    ----------
    string_choices : List[str]
        List of string choices to fit the predictor on.
    """

    # Create a predictor with default parameters.
    predictor = FuzzyPredictor()
    
    # Fit the predictor with the string choices.
    result = predictor.fit(string_choices)
    
    # Check that fit sets the internal state correctly.
    assert (
        predictor.raw_documents == string_choices
    ), "raw_documents should be set to the fitted data."

    assert (
        predictor._is_fitted is True
    ), "Predictor should be marked as fitted."
    
    # Check that fit returns self for method chaining.
    assert (
        result is predictor
    ), "fit should return self for method chaining."

def test_fuzzy_predictor_transform(string_choices: List[str]) -> None:
    """
    Test transform method of FuzzyPredictor.
    
    This test verifies that the transform method correctly returns the
    input data unchanged and raises an error if the predictor is not
    fitted.
    
    Parameters
    ----------
    string_choices : List[str]
        List of string choices to transform.
    """

    # Create a predictor.
    predictor = FuzzyPredictor()
    
    # Fit the predictor.
    predictor.fit(string_choices)
    
    # Transform should return the input data unchanged for
    # FuzzyPredictor.
    result = predictor.transform(string_choices)
    assert (
        result == string_choices
    ), "transform should return the input data unchanged."
    
    # Check that transform raises error if not fitted.
    new_predictor = FuzzyPredictor()

    # Transform should raise an error if not fitted.
    with pytest.raises(ValueError):
        new_predictor.transform(string_choices)

def test_fuzzy_predictor_fit_transform(string_choices: List[str]) -> None:
    """
    Test fit_transform method of FuzzyPredictor.
    
    This test verifies that the fit_transform method correctly fits the
    model and returns the input data unchanged.
    
    Parameters
    ----------
    string_choices : List[str]
        List of string choices to fit and transform.
    """

    # Create a predictor with default parameters.
    predictor = FuzzyPredictor()
    
    # fit_transform should fit the model and return the input data
    # unchanged.
    result = predictor.fit_transform(string_choices)
    
    # Check internal state after fit_transform.
    assert (
        predictor.raw_documents == string_choices
    ), "raw_documents should be set to the fitted data."

    assert (
        predictor._is_fitted is True
    ), "Predictor should be marked as fitted."
    
    # Check result of fit_transform.
    assert (
        result == string_choices
    ), "fit_transform should return the input data unchanged."

########################################################################
## PREDICTION TESTS
########################################################################

def test_fuzzy_predictor_predict_single_query(
        string_choices: List[str]
    ) -> None:
    """
    Test predict method with a single query.
    
    This test verifies that the predict method correctly returns a match
    for a single query string.
    
    Parameters
    ----------
    string_choices : List[str]
        List of string choices to fit the predictor on.
    """

    # Create a predictor.
    predictor = FuzzyPredictor()

    # Fit the predictor.
    predictor.fit(string_choices)
    
    # Predict with a single query that is slightly misspelled.
    query = "aple"
    result = predictor.predict(query)
    
    # Verify the result format and content.
    assert (
        isinstance(result, list)
    ), "Prediction result should be a list."

    assert (
        len(result) == 1
    ), "Result should contain one item for a single query."

    assert (
        isinstance(result[0], str)
    ), "Each prediction should be a string."

    assert (
        result[0] in string_choices
    ), "Prediction should be one of the choices."

def test_fuzzy_predictor_predict_multiple_queries(
        string_choices: List[str]
    ) -> None:
    """
    Test predict method with multiple queries.
    
    This test verifies that the predict method correctly returns matches
    for a list of query strings.
    
    Parameters
    ----------
    string_choices : List[str]
        List of string choices to fit the predictor on.
    """

    # Create a predictor.
    predictor = FuzzyPredictor()

    # Fit the predictor.
    predictor.fit(string_choices)
    
    # Predict with multiple queries that are slightly misspelled.
    queries = ["aple", "ornge"]
    result = predictor.predict(queries)
    
    # Verify the result format and content.
    assert (
        isinstance(result, list)
    ), "Prediction result should be a list."

    assert (
        len(result) == len(queries)
    ), "Result should contain one item per query."
    
    # Verify each prediction.
    for i, query in enumerate(queries):
        assert (
            isinstance(result[i], str)
        ), "Each prediction should be a string."

        assert (
            result[i] in string_choices
        ), "Each prediction should be one of the choices."

def test_fuzzy_predictor_predict_with_return_score(
        string_choices: List[str]
    ) -> None:
    """
    Test predict method with return_score=True.
    
    This test verifies that the predict method correctly returns matches
    with scores when return_score is set to True.
    
    Parameters
    ----------
    string_choices : List[str]
        List of string choices to fit the predictor on.
    """

    # Create a predictor.
    predictor = FuzzyPredictor()

    # Fit the predictor.
    predictor.fit(string_choices)
    
    # Predict with return_score=True to get scores with matches.
    query = "aple"
    result = predictor.predict(query, return_score=True)
    
    # Verify the result format and content.
    assert (
        isinstance(result, list)
    ), "Prediction result should be a list."

    assert (
        len(result) == 1
    ), "Result should contain one item for a single query."

    assert (
        isinstance(result[0], str)
    ), "Each prediction should be a string."

    assert (
        result[0] in string_choices
    ), "Prediction should be one of the choices."

def test_fuzzy_predictor_predict_without_fitting() -> None:
    """
    Test predict method without first fitting the model.
    
    This test verifies that the predict method raises a ValueError
    if called before the model is fitted.
    """

    # Create a predictor but don't fit it.
    predictor = FuzzyPredictor()
    
    # Predict without fitting should raise an error.
    with pytest.raises(ValueError):
        predictor.predict("apple")

def test_fuzzy_predictor_predict_with_custom_limit(
        string_choices: List[str]
    ) -> None:
    """
    Test predict method with custom limit parameter.
    
    This test verifies that the predictor correctly respects the limit
    parameter for the number of results to return.
    
    Parameters
    ----------
    string_choices : List[str]
        List of string choices to fit the predictor on.
    """
    # Create a predictor with a custom limit of 2.
    predictor = FuzzyPredictor(limit=2)
    predictor.fit(string_choices)
    
    # Get the top 2 matches.
    query = "aple"
    result = predictor.predict_proba(query)
    
    # Verify the result format and content.
    assert (
        isinstance(result, list)
    ), "Prediction result should be a list."

    assert (
        len(result) == 1
    ), "Result should contain one list for a single query."

    assert (
        len(result[0]) == 2
    ), "Result should contain exactly 2 matches when limit=2."
    
    # Verify each prediction tuple.
    for r in result[0]:
        assert (
            isinstance(r, tuple)
        ), "Each prediction should be a tuple."

        assert (
            len(r) == 2
        ), "Each prediction tuple should have 2 elements (choice, score)."

        assert (
            r[0] in string_choices
        ), "First element should be one of the choices."

        assert (
            isinstance(r[1], float)
        ), "Second element should be a float score."

def test_fuzzy_predictor_predict_with_unlimited_results(
        string_choices: List[str]
    ) -> None:
    """
    Test predict method with no result limit.
    
    This test verifies that the predictor correctly returns all matches
    when limit is set to None.
    
    Parameters
    ----------
    string_choices : List[str]
        List of string choices to fit the predictor on.
    """

    # Create a predictor with limit=None (no limit).
    predictor = FuzzyPredictor(limit=None)
    predictor.fit(string_choices)
    
    # Get all matches.
    query = "aple"
    result = predictor.predict_proba(query)
    
    # Verify the result format and content.
    assert (
        isinstance(result, list)
    ), "Prediction result should be a list."

    assert (
        len(result) == 1
    ), "Result should contain one list for a single query."

    assert (
        len(result[0]) == len(string_choices)
    ), "Result should contain all choices when limit=None."
    
    # Verify each prediction tuple.
    for r in result[0]:
        assert (
            isinstance(r, tuple)
        ), "Each prediction should be a tuple."

        assert (
            len(r) == 2
        ), "Each prediction tuple should have 2 elements (choice, score)."

        assert (
            r[0] in string_choices
        ), "First element should be one of the choices."

        assert (
            isinstance(r[1], float)
        ), "Second element should be a float score."

########################################################################
## PREDICT_PROBA TESTS
########################################################################

def test_fuzzy_predictor_predict_proba_single_query(
        string_choices: List[str]
    ) -> None:
    """
    Test predict_proba method with a single query.
    
    This test verifies that the predict_proba method correctly returns
    matches with scores for a single query.
    
    Parameters
    ----------
    string_choices : List[str]
        List of string choices to fit the predictor on.
    """

    # Create a predictor.
    predictor = FuzzyPredictor()

    # Fit the predictor.
    predictor.fit(string_choices)
    
    # Get probability scores for a single query.
    query = "aple"
    result = predictor.predict_proba(query)
    
    # Verify the result format and content.
    assert (
        isinstance(result, list)
    ), "Prediction result should be a list."

    assert (
        len(result) == 1
    ), "Result should contain one list for a single query."

    assert (
        isinstance(result[0], list)
    ), "Each query result should be a list of scores."
    
    # Check that the number of results is limited by the predictor's
    # limit.
    assert (
        len(result[0]) <= predictor.limit
    ), "Number of results should be limited by predictor.limit."
    
    # Verify each score tuple.
    for choice_score in result[0]:
        assert (
            isinstance(choice_score, tuple)
        ), "Each score should be a tuple."

        assert (
            len(choice_score) == 2
        ), "Each score tuple should have 2 elements (choice, score)."

        assert (
            choice_score[0] in string_choices
        ), "First element should be one of the choices."

        assert (
            isinstance(choice_score[1], float)
        ), "Second element should be a float score."

def test_fuzzy_predictor_predict_proba_multiple_queries(
        string_choices: List[str]
    ) -> None:
    """
    Test predict_proba method with multiple queries.
    
    This test verifies that the predict_proba method correctly returns
    matches with scores for multiple queries.
    
    Parameters
    ----------
    string_choices : List[str]
        List of string choices to fit the predictor on.
    """

    # Create a predictor.
    predictor = FuzzyPredictor()

    # Fit the predictor.
    predictor.fit(string_choices)
    
    # Get probability scores for multiple queries.
    queries = ["aple", "ornge"]
    result = predictor.predict_proba(queries)
    
    # Verify the result format and content.
    assert (
        isinstance(result, list)
    ), "Prediction result should be a list."

    assert (
        len(result) == len(queries)
    ), "Result should contain one list per query."
    
    # Verify each query's results.
    for query_result in result:

        # Each query result should be a list of (choice, score) tuples.
        assert (
            isinstance(query_result, list)
        ), "Each query result should be a list."
        
        # Check that the number of results is limited by the predictor's
        # limit.
        assert (
            len(query_result) <= predictor.limit
        ), "Number of results should be limited by predictor.limit."
        
        # Verify each score tuple.
        for choice_score in query_result:
            assert (
                isinstance(choice_score, tuple)
            ), "Each score should be a tuple."

            assert (
                len(choice_score) == 2
            ), "Each score tuple should have 2 elements (choice, score)."

            assert (
                choice_score[0] in string_choices
            ), "First element should be one of the choices."

            assert (
                isinstance(choice_score[1], float)
            ), "Second element should be a float score."

########################################################################
## ALGORITHMS AND SCORING TESTS
########################################################################

def test_fuzzy_predictor_levenshtein_distance(
        string_choices: List[str]
    ) -> None:
    """
    Test FuzzyPredictor with Levenshtein distance algorithm.
    
    This test verifies that the Levenshtein distance algorithm correctly
    matches misspelled words to their closest matches.
    
    Parameters
    ----------
    string_choices : List[str]
        List of string choices to fit the predictor on.
    """

    # Create a predictor.
    predictor = FuzzyPredictor(algorithm="levenshtein", method="distance")

    # Fit the predictor.
    predictor.fit(string_choices)
    
    # "aple" should match "apple" with Levenshtein distance of 1.
    result = predictor.predict("aple")
    assert (
        result[0][1] == "apple"
    ), "The word 'aple' should match 'apple' with Levenshtein distance."

def test_fuzzy_predictor_jaro_similarity(string_choices: List[str]) -> None:
    """
    Test FuzzyPredictor with Jaro similarity algorithm.
    
    This test verifies that the Jaro similarity algorithm correctly
    matches misspelled words to their closest matches.
    
    Parameters
    ----------
    string_choices : List[str]
        List of string choices to fit the predictor on.
    """

    # Create a predictor.
    predictor = FuzzyPredictor(algorithm="jaro", method="similarity")

    # Fit the predictor.
    predictor.fit(string_choices)
    
    # "pinapple" should match "pineapple" with Jaro similarity.
    result = predictor.predict("pinapple")
    
    assert (
        result[0][1] == "pineapple"
    ), "The word 'pinapple' should match 'pineapple' with Jaro similarity."

def test_fuzzy_predictor_score_cutoff(string_choices: List[str]) -> None:
    """
    Test FuzzyPredictor with score_cutoff parameter.
    
    This test verifies that the score_cutoff parameter correctly
    filters matches based on their scores.
    
    Parameters
    ----------
    string_choices : List[str]
        List of string choices to fit the predictor on.
    """

    # Create a predictor.
    predictor = FuzzyPredictor(
        algorithm="levenshtein", 
        method="distance",
        score_cutoff=1  # Only exact matches or matches with distance <= 1.
    )

    # Fit the predictor.
    predictor.fit(string_choices)
    
    # "bananas" is 1 edit away from "banana", should match.
    result = predictor.predict_proba("bananas")
    assert (
        len(result[0]) > 0
    ), "The word 'bananas' should match at least one choice with distance <= 1."
    
    # "bananarama" is more than 1 edit away from any choice, should
    # return limited results.
    result = predictor.predict_proba("bananarama")
    assert (
        len(result[0]) < len(string_choices)
    ), "The word 'bananarama' should match fewer choices due to the cutoff."

########################################################################
## HELPER METHOD TESTS
########################################################################

def test_fuzzy_predictor_check_is_fitted() -> None:
    """
    Test _check_is_fitted method.
    
    This test verifies that the _check_is_fitted method correctly
    raises an error if the predictor is not fitted.
    """
    
    # Create a predictor but don't fit it.
    predictor = FuzzyPredictor()
    
    # Check is_fitted should raise error if not fitted.
    with pytest.raises(ValueError):
        predictor._check_is_fitted()
    
    # Mark as fitted and check again.
    predictor._is_fitted = True
    predictor._check_is_fitted()