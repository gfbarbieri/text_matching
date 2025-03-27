"""
Fuzzy text matching module using RapidFuzz.

This module provides a fuzzy text matching predictor that uses various
string similarity algorithms from the RapidFuzz library. It supports
both supervised and unsupervised text matching tasks.

The predictor can be used in two modes:
1. Supervised: Match text to known reference labels
2. Unsupervised: Find similar documents within a corpus
"""

########################################################################
## IMPORTS
########################################################################

import numpy as np

from rapidfuzz.process import cdist
from rapidfuzz.utils import default_process

from typing import List, Tuple, Dict, Optional, Union, Callable, Any

from text_prediction.predictors.base.base import BasePredictor
from text_prediction.predictors.distance import ALGORITHMS, METHODS

########################################################################
## FUZZY SEARCH CLASS
########################################################################

class FuzzyPredictor(BasePredictor):
    """
    A class for fuzzy text matching using RapidFuzz.

    This predictor uses various string similarity algorithms to find
    matches between text documents. It supports both supervised matching
    (to known labels) and unsupervised matching (within a corpus).

    Parameters
    ----------
    algorithm : str, default="levenshtein"
        The comparison algorithm to use. Must be one of:
        "damerau_levenshtein", "hamming", "indel", "jaro",
        "jaro_winkler", "levenshtein".
    method : str, default="distance"
        The method to call on the algorithm. Must be one of:
        "distance", "normalized_distance", "similarity",
        "normalized_similarity".
    processor : callable, default=default_process
        A function to preprocess the query and choices.
    limit : int, default=5
        Maximum number of results to return.
    score_cutoff : float, optional
        A filter threshold. For distance methods (lower is better),
        only results with scores <= threshold are kept. For
        similarity methods (higher is better), only results with
        scores >= threshold are kept.
    score_hint : float, optional
        A hint for the score cutoff.
    scorer_kwargs : dict, optional
        Additional keyword arguments to pass to the scorer.
    score_multiplier : float, default=1
        A multiplier for the score.
    dtype : dtype, optional
        The data type of the output array.
    workers : int, default=1
        The number of workers to use for parallel processing.

    Attributes
    ----------
    algorithms : dict
        A dictionary mapping algorithm names to their RapidFuzz classes.
    valid_methods : set
        A set of valid methods for the fuzzy search.
    raw_documents : list[str]
        The reference documents to match against. In supervised mode,
        these are the unique labels. In unsupervised mode, these are
        the documents to match within.

    Examples
    --------

    Supervised matching (to known labels):

    .. code-block:: python

       from text_prediction.predictors.distance import FuzzyPredictor
       
       predictor = FuzzyPredictor(algorithm="levenshtein", method="similarity")
       predictor.fit(X, y)  # y contains reference labels
       matches = predictor.predict(new_text)

    Unsupervised matching (within corpus):

    .. code-block:: python

       predictor = FuzzyPredictor(algorithm="jaro", method="normalized_similarity")
       predictor.fit(documents)  # documents to match within
       matches = predictor.predict(new_documents)
    """

    def __init__(
        self,
        algorithm: str = "levenshtein",
        method: str = "distance",
        processor: Callable = default_process,
        limit: int = 5,
        score_cutoff: Optional[float] = None,
        score_hint: Optional[float] = None,
        scorer_kwargs: Optional[Dict] = None,
        score_multiplier: float = 1,
        dtype: Any = None,
        workers: int = 1
    ) -> None:
        """
        Initialize the FuzzySearch instance.

        Parameters
        ----------
        algorithm : str, default="levenshtein"
            The comparison algorithm to use.
        method : str, default="distance"
            The method to call on the algorithm.
        processor : callable, default=default_process
            A function to preprocess the query and choices.
        limit : int, default=5
            Maximum number of results to return.
        score_cutoff : float, optional
            A filter threshold.
        score_hint : float, optional
            A hint for the score cutoff.
        scorer_kwargs : dict, optional
            Additional keyword arguments to pass to the scorer.
        score_multiplier : float, default=1
            A multiplier for the score.
        dtype : dtype, optional
            The data type of the output array.
        workers : int, default=1
            The number of workers to use for parallel processing.
        """

        # Initialize the attributes.
        self.algorithm = algorithm
        self.method = method
        self.processor = processor
        self.limit = limit
        self.score_cutoff = score_cutoff
        self.score_hint = score_hint
        self.scorer_kwargs = scorer_kwargs
        self.score_multiplier = score_multiplier
        self.dtype = dtype
        self.workers = workers
        
        # Set up algorithms mapping.
        self.scorer = self._get_scorer(algorithm=algorithm, method=method)

        # Will be set during fit.
        self.raw_documents = None
        self._is_fitted = False

    def fit(self, X: List[str], y=None) -> 'FuzzyPredictor':
        """
        Fit the FuzzySearch instance with the list of choices.

        Parameters
        ----------
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Raw documents to process. In supervised mode (y provided), 
            these are queries. In unsupervised mode (no y), these are 
            the reference documents to match against.
        y : array-like of shape (n_samples,), optional
            Target labels. If provided, builds vocabulary from these labels
            (supervised mode). If None, builds vocabulary from X 
            (unsupervised mode).

        Returns
        -------
        self : FuzzySearch
            The fitted instance.

        Notes
        -----
        This method is only present for API consistency with
        scikit-learn. In FuzzySearch, we don't fit the input data.
        """

        # Store the raw documents.
        if y is not None:
            # Supervised case: build vocab from unique labels.
            self.raw_documents = np.unique(y)
        else:
            # Unsupervised case: build vocab from X.
            self.raw_documents = X
        
        # Set the model as fitted.
        self._is_fitted = True
        
        return self

    def transform(self, X: List[str]) -> List[str]:
        """
        Transform method for scikit-learn API compatibility.
        
        For FuzzySearch, this is an identity operation as the actual
        text processing happens internally during the prediction phase.

        Parameters
        ----------
        X : List[str]
            Documents to transform.

        Returns
        -------
        List[str]
            The same documents, unchanged.

        Notes:
        ------
        This method is only present for API consistency with
        scikit-learn. In FuzzySearch, we don't transform the input data.
        Processing happens internally during prediction using the
        processor function.
        """

        self._check_is_fitted()
        
        return X

    def fit_transform(self, X: List[str], y=None) -> List[str]:
        """
        Fit the FuzzySearch instance and transform the input documents.

        Parameters
        ----------
        raw_documents : List[str]
            The list of strings to search within.
        y : ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        List[str]
            Transformed documents.
        """

        self.fit(X=X, y=y)
        
        return self.transform(X=X)

    def predict(self, X: Union[str, List[str]]) -> List[Tuple[str, str]]:
        """
        Predict the best matches for each query.

        Parameters
        ----------
        X : str or List[str]
            The query string or list of query strings.

        Returns
        -------
        List[Tuple[str, str]]
            A list of best matches.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """

        # Check if the model is fitted.
        self._check_is_fitted()

        # Validate the query. This will convert a single query to a
        # list if it's a string.
        X = self._validate_query(X)

        # Get predictions with scores. X is always converted to a list.
        # If X is a single query, then the scores will have a shape of
        # (1, n_choices), where each element is a tuple of (choice,
        # score).
        # 
        # If X is a list of queries, then the scores will have a shape
        # of (n_queries, n_choices), where each element is a tuple of
        # (choice, score).
        scores = self.predict_proba(X)

        # The predict will return the best prediction, ie, the best
        # match for each query, without the scores.
        # 
        # If X is searching for a single string, then the scores array
        # will return a list of tuples (n_choices,), where each element 
        # is a tuple (choice, score). If X is searching for multiple
        # strings, then the scores array will return a list of lists of
        # tuples (n_queries, n_choices), where each element is a tuple
        # (choice, score).
        #
        # If the length of X is 1, then we are expecting a single list
        # of tuples (choice, score). The best match is already sorted
        # to be the first element.
        #
        # If the length of X is greater than 1, then we are expecting
        # a list of lists of tuples (choice, score).
        results = []

        if len(X) == 1:
            results = scores[0]
        else:
            results = [score[0][0] for score in scores]

        return results

    def predict_proba(
            self, X: Union[str, List[str]]
        ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """
        Return similarity or distance scores for each possible outcome.

        Parameters
        ----------
        X : str or List[str]
            The query string or list of query strings.

        Returns
        -------
        List[Tuple[str, float]] or List[List[Tuple[str, float]]]
            For a single query: a list of (match, score) tuples.
            For multiple queries: a list of lists, where each inner list 
            contains (match, score) tuples for that query.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """

        # Check if the model is fitted.
        self._check_is_fitted()
        
        # Validate the query. This will convert a single query to a
        # list if it's a string.
        X = self._validate_query(X)

        # cdist will return a matrix of scores, where each row is a
        # query and each column is a choice. The elements are the
        # scores. This will return a shape (n_queries, n_choices) array.
        # The column index corresponds to the choice index.

        scorer_kwargs = self.scorer_kwargs if self.scorer_kwargs else {}

        scores = cdist(
            queries=X, choices=self.raw_documents, scorer=self.scorer,
            processor=self.processor, score_cutoff=self.score_cutoff,
            score_hint=self.score_hint, scorer_kwargs=scorer_kwargs,
            score_multiplier=self.score_multiplier, dtype=self.dtype,
            workers=self.workers
        )
        
        # Sort based on method (ascending for distance, descending for
        # similarity). The argsort function returns the indices that
        # would sort the array. Axis=1 means sort along the columns
        # (choices).
        # 
        # For example, if we have two queries and three choices, the
        # scores matrix is 2x3. The argsort(scores, axis=1) will return
        # a 2x3 matrix of indices that would sort the scores matrix for
        # each query independently. The [:, ::-1] is used to reverse the
        # order of the indices for similarity methods.
        if self.scorer.__name__ in ["distance", "normalized_distance"]:
            sorted_indices = np.argsort(scores, axis=1)
        elif self.scorer.__name__ in [
                "similarity", "normalized_similarity"
            ]:
            sorted_indices = np.argsort(scores, axis=1)[:, ::-1]
        
        # Apply limit to the sorted indices.
        if self.limit:
            sorted_indices = sorted_indices[:, :self.limit]

        # Now, we need to get back the choice values and scores
        # using the sorted indices. We are going to loop over the
        # the sorted score indices which correspond to both the
        # choices (self.raw_documents) and the scores (scores).
        #
        # The choices have shape (n_choices,) and the query has
        # shape (n_queries,). The scores have shape (n_queries,
        # n_choices) and the sorted indices also have shape
        # (n_queries, n_choices).
        results = []

        # Loop over the sorted score indices. The i indexes the
        # query (row) and score_indices are the sorted indices we
        # want to use to get the choices and scores.
        for i, score_indices in enumerate(sorted_indices):

            # Get the choice and score for each sorted index.
            # self.raw_documents has the choices and scores has the
            # scores.
            choice_scores = [
                (self.raw_documents[idx], scores[i, idx])
                for idx in score_indices
            ]

            # Append the choice and score tuples to the results.
            results.append(choice_scores)

        return results

    def _get_scorer(self, algorithm: str, method: str) -> Callable:
        """
        Get the scorer.

        Returns
        -------
        Callable
            The scorer.
        """

        # Validate the algorithm and method. It's better to do this here
        # than in init because the user may change the algorithm or
        # method after initialization.
        algorithm = self._validate_algorithm(algorithm)
        method = self._validate_method(method)

        # Get the algorithm class and method.
        scorer = getattr(ALGORITHMS[algorithm], method)

        return scorer

    def _check_is_fitted(self) -> None:
        """
        Check if the model is fitted.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """

        if not self._is_fitted or self.raw_documents is None:
            raise ValueError(
                "This FuzzySearch instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using "
                "this estimator."
            )
    
    def _validate_algorithm(self, algorithm: str) -> str:
        """
        Validate the algorithm.

        Parameters
        ----------
        algorithm : str
            The algorithm to validate.

        Returns
        -------
        str
            The validated algorithm.
            
        Raises
        ------
        ValueError
            If the algorithm is not supported.
        """

        # Validate the algorithm.
        if algorithm not in ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}. "
                f"Supported algorithms: {ALGORITHMS}"
            )

        return algorithm

    def _validate_method(self, method: str) -> str:
        """
        Validate the method.

        Parameters
        ----------
        method : str
            The method to validate.

        Returns
        -------
        str
            The validated method.
            
        Raises
        ------
        ValueError
            If the method is not supported.
        """

        # Validate the method.
        if method not in METHODS:
            raise ValueError(
                f"Unsupported method: {method}. "
                f"Supported methods: {METHODS}"
            )

        return method
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """

        # Get parameters from the vectorizer.
        params = {
            'algorithm': self.algorithm,
            'method': self.method,
            'processor': self.processor,
            'limit': self.limit,
            'score_cutoff': self.score_cutoff,
            'score_hint': self.score_hint,
            'scorer_kwargs': self.scorer_kwargs,
            'score_multiplier': self.score_multiplier,
            'dtype': self.dtype,
        }

        return params
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
            
        Returns
        -------
        self : object
            Returns self.
        """

        # Set parameters for the estimator.
        for param, value in params.items():

            # Set the parameter.
            setattr(self, param, value)

            # Update scorer if its parameters change.
            if param in ['algorithm', 'method']:

                # Get the scorer and create a new instance with the new
                # parameters.
                self.scorer = self._get_scorer(
                    algorithm=self.algorithm, method=self.method
                )

        return self