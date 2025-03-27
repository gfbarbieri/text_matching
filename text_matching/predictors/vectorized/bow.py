"""
Bag-of-words text matching module.

This module provides a bag-of-words based text matching predictor that
can be used for both supervised and unsupervised text matching tasks.
It supports various text vectorization methods and similarity metrics.

The predictor can be used in two modes:
1. Supervised: Match text to known reference labels
2. Unsupervised: Find similar documents within a corpus
"""

########################################################################
## IMPORTS
########################################################################

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from typing import Callable

from text_matching.predictors.base.base import BasePredictor

########################################################################
## BAG OF WORDS (BOW) PREDICTOR CLASS
########################################################################

class BOWPredictor(BasePredictor):
    """
    A class for text matching using bag-of-words approaches.

    This predictor uses text vectorization and similarity metrics to
    find matches between text documents. It supports both supervised
    matching (to known labels) and unsupervised matching (within a
    corpus).

    Parameters
    ----------
    vectorizer : text feature extractor class, optional
        A text feature extractor class to use for vectorization. Default
        is CountVectorizer. Must be a class, not an instance.
    metric : similarity metric, optional
        The metric to use for similarity search. Default is cosine
        similarity.
    analyzer : string, optional
        Type of analyzer for text vectorization (default 'word').
    ngram_range : tuple, optional
        The ngram range for vectorization. Default is (1, 1) for
        traditional bag of words.
    limit : int or None, optional
        Number of top matches to return. If None, returns all
        matches. Default is 1.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the text vectorizer.

    Attributes
    ----------
    metric : similarity metric
        The metric to use for similarity search.
    raw_documents : list of strings
        The reference documents to match against. In supervised mode,
        these are the unique labels. In unsupervised mode, these are
        the documents to match within.
    vectorizer : text feature extractor
        The vectorizer for converting text to vectors.
    vectors : array-like
        The vectorized reference documents.

    Examples
    --------
    Supervised matching (to known labels):

    .. code-block:: python

       from text_matching.predictors.vectorized import BOWPredictor

       predictor = BOWPredictor(analyzer='char_wb', ngram_range=(2,3))
       predictor.fit(X, y)  # y contains reference labels.
       matches = predictor.predict(new_text)

    Unsupervised matching (within corpus):

    .. code-block:: python

       predictor = BOWPredictor(analyzer='char_wb', ngram_range=(2,3))
       predictor.fit(documents)  # documents to match within.
       matches = predictor.predict(new_documents)

    """

    def __init__(
            self, vectorizer: Callable | None=None,
            metric: Callable | None=None, analyzer='word', ngram_range=(1, 1),
            limit: int | None=None, **kwargs
        ) -> None:
        """
        Initialize the matcher with a list of store names.
        
        Parameters
        ----------
        vectorizer : text feature extractor, optional
            A text feature extractor class (not a class instance) to use
            for vectorization. Default is a CountVectorizer.
        metric : similarity metric, optional
            The metric to use for similarity search. Default is cosine
            similarity.
        analyzer : string, optional
            Type of analyzer for CountVectorizer. Default is a
            word-based analyzer (default 'word').
        ngram_range : tuple, optional
            The ngram range for vectorization. Default is a traditional
            bag of words implementation with ngrams of length 1.
        limit : int, optional
            The number of top matches to return. Default is 1.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the text vectorizer.
        """

        # Initialize the attributes.
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.limit = limit

        # Initialize the vectorizer and fit it on the choices. This is
        # the underlying model that will be used to transform the query
        # and choices into a vector space.
        if vectorizer is None:
            self.vectorizer = CountVectorizer(
                analyzer=analyzer, ngram_range=ngram_range, **kwargs
            )
        else:
            self.vectorizer = vectorizer(
                analyzer=analyzer, ngram_range=ngram_range, **kwargs
            )

        # Initialize the metric.
        if metric is None:
            self.metric = cosine_similarity
        else:
            self.metric = metric

        # Initialize the raw documents.
        self.raw_documents = None

    def fit(self, X, y=None):
        """
        Fit the vectorizer on reference documents.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Raw documents to process. In supervised mode (y provided), 
            these are queries. In unsupervised mode (no y), these are 
            the reference documents to match against.
        y : array-like of shape (n_samples,), optional
            Target labels. If provided, builds vocabulary from these
            labels (supervised mode). If None, builds vocabulary from
            X (unsupervised mode).

        Returns
        -------
        self : BOWPredictor
            Fitted predictor.
        """

        # Store the raw documents.
        if y is not None:
            # Supervised case: build vocab from unique labels.
            self.raw_documents = np.unique(y)
        else:
            # Unsupervised case: build vocab from X.
            self.raw_documents = X
        
        # Fit vectorizer on reference documents.
        self.vectorizer.fit(self.raw_documents)

        # Store the vectorized raw documents.
        self.vectors = self.vectorizer.transform(self.raw_documents)
        
        return self

    def transform(self, X: list[str]) -> 'np.ndarray':
        """
        Transform the provided documents into a vector space.

        Parameters
        ----------
        X : list of strings
            The documents to transform into a vector space.

        Returns
        -------
        np.ndarray
            The transformed documents in vector space.
        """

        # Store the vectorized raw documents.
        vectors = self.vectorizer.transform(X)

        # Return the vectorized raw documents.
        return vectors

    def fit_transform(self, X: list[str], y=None) -> 'np.ndarray':
        """
        Fit the vectorizer on the provided documents and transform them
        into a vector space.

        Parameters
        ----------
        X : list of strings
            The documents to fit the vectorizer on and transform into a
            vector space.
        y : ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        np.ndarray
            The transformed documents in vector space.
        """

        # Fit the vectorizer on the provided documents.
        self.fit(X=X, y=y)

        return self.vectors

    def predict(
            self, X: str | list[str]
        ) -> list[tuple[str, str, float]] | list[tuple[str, str]]:
        """
        Search for the top N best matching store names for the given
        query or queries.
        
        Parameters
        ----------
        raw_documents : string or list of strings
            The user-provided query or list of queries.
        
        Returns
        -------
        results : list
            List of tuples (query, choice, score) if return_score is
            True, or (query, choice) if return_score is False. For
            single queries, query will be the same for all results.
        """

        # If limit is None, set it to the number of choices.
        if self.limit is None:
            self.limit = len(X)

        # Validate the query.
        X = self._validate_query(X)

        # Compute cosine similarities between the queries and the
        # raw documents in vector space. Predict_proba() returns a
        # list of lists, where each inner list contains a tuple of
        # (choice, score) for each query.
        scores = self.predict_proba(X=X)
        
        # Return the best choice for each query. The first slicer
        # returns the choice with the highest score, and the second
        # slicer returns only the choice text.
        results = [score[0][0] for score in scores]
    
        return results
    
    def predict_proba(
            self, X: str | list[str]
        ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]:
        """
        Return similarity or distance scores for each possible outcome.

        Parameters
        ----------
        X : str or list[str]
            The query string or list of query strings.

        Returns
        -------
        list[tuple[str, float]] or list[list[tuple[str, float]]]
            For a single query: a list of (match, score) tuples.
            For multiple queries: a list of lists, where each inner list 
            contains (match, score) tuples for that query.

        Raises
        ------
        ValueError
            If the model has not been fitted.
        """

        # If limit is None, set it to the number of choices.
        if self.limit is None:
            self.limit = len(X)

        # Validate the query.
        X = self._validate_query(X)
        
        # Use the fitted model to transform the queries into the same
        # vector space as the raw documents.
        X_vectors = self.transform(X)

        # Compute cosine similarities between the queries and the
        # raw documents in vector space.
        similarities = self.metric(X_vectors, self.vectors)

        # Get indices of top N matches for each query. Argsort() is a
        # function that returns the indices that would sort an array in
        # ascending order.
        # 
        # The first slicer reverses asks to return all rows (:), but to
        # reverse the order the list (::-1) such that the match with the
        # highest similarity is at the top. The second slicer asks to
        # return all rows (:), then to return only the limit matches
        # (:limit). This is applied to each row (query) in the
        # similarities array.
        #
        # For example, if we have 2 queries and 4 choices, the
        # similarities array is 2x4. The top_indices array will be 2x4.
        # similarities = np.array([
        #     [0.1, 0.8, 0.3, 0.9],  # query 1
        #     [0.4, 0.2, 0.7, 0.5]   # query 2
        # ])
        #
        # similarities.argsort() gives:
        # [[0, 2, 1, 3],  # indices for query 1
        #  [1, 0, 3, 2]]  # indices for query 2
        #
        # similarities.argsort()[:, ::-1] gives:
        # [[3, 1, 2, 0],  # reversed for query 1
        #  [2, 3, 0, 1]]  # reversed for query 2
        #
        # similarities.argsort()[:, ::-1][:, :limit] for limit=2:
        # [[3, 1],  # top 2 indices for query 1
        #  [2, 3]]  # top 2 indices for query 2
        top_indices = similarities.argsort()[:, ::-1][:, :self.limit]
        
        # For each query (i) and comparison (j), get the matched choice
        # and similarity value. Query is 1xn; comparisons are nx1; and
        # similarities are nxn.
        results = []

        for i, indices in enumerate(top_indices):

            # Get matched choice and similarity value for this query in
            # format (choice, score).
            matches = [
                (self.raw_documents[j], similarities[i, j]) for j in indices
            ]
            
            # Extend the results with the current query's results.
            results.append(matches)

        return results
    
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
            'analyzer': self.analyzer,
            'ngram_range': self.ngram_range
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

            # Update vectorizer if its parameters change.
            if param in ['analyzer', 'ngram_range']:

                # Get the vectorizer class and create a new instance
                # with the new parameters.
                self.vectorizer = self.vectorizer.__class__(
                    analyzer=self.analyzer,
                    ngram_range=self.ngram_range
                )

        return self