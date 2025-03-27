"""
Base module for text matching predictors.

This module defines the base predictor class that all text matching
predictors must inherit from. It ensures a consistent interface across
all predictors, making them compatible with scikit-learn's API.

The base class supports both supervised and unsupervised text matching:
- Supervised: Match text to known reference labels
- Unsupervised: Find similar documents within a corpus
"""

########################################################################
## IMPORTS
########################################################################

from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Any

########################################################################
## BASE PREDICTOR CLASS
########################################################################

class BasePredictor(ABC):
    """
    Base class for all text matching predictors.
    
    This class defines the interface that all text matching predictors
    must implement. It ensures a consistent API across all predictors
    and compatibility with scikit-learn's API.

    The base class supports both supervised and unsupervised text matching:
    - Supervised: Match text to known reference labels (y)
    - Unsupervised: Find similar documents within a corpus (X)

    Attributes
    ----------
    raw_documents : List[str], optional
        The reference documents after fitting. In supervised mode, these
        are the unique labels. In unsupervised mode, these are the
        documents to match against.
    _is_fitted : bool
        Whether the predictor has been fitted.
    """
    
    def __init__(self) -> None:
        """
        Initialize the predictor.
        """

        # Will be set during fit.
        self.raw_documents = None
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, raw_documents: List[str], y=None) -> 'BasePredictor':
        """
        Fit the predictor on the provided raw documents.
        
        Parameters
        ----------
        raw_documents : List[str]
            The raw documents to fit the predictor on.
        y : Any, optional
            Target values. Included for scikit-learn compatibility but
            not used.
            
        Returns
        -------
        BasePredictor
            The fitted predictor (self).
        """

        pass
    
    @abstractmethod
    def transform(self, raw_documents: List[str]) -> Any:
        """
        Transform the provided raw documents.
        
        Parameters
        ----------
        raw_documents : List[str]
            The raw documents to transform.
            
        Returns
        -------
        Any
            The transformed documents. Type depends on the specific
            predictor.
        """

        pass
    
    @abstractmethod
    def fit_transform(self, raw_documents: List[str], y=None) -> Any:
        """
        Fit the predictor on the provided raw documents and transform
        them.
        
        Parameters
        ----------
        raw_documents : List[str]
            The raw documents to fit on and transform.
        y : Any, optional
            Target values. Included for scikit-learn compatibility but
            not used.
            
        Returns
        -------
        Any
            The transformed documents. Type depends on the specific
            predictor.
        """

        pass
    
    @abstractmethod
    def predict(self, X: Union[str, List[str]]) -> List[str]:
        """
        Predict matches for the given queries.
        
        Parameters
        ----------
        X : str or List[str]
            The query or queries to predict matches for.
            
        Returns
        -------
        List[str]
            List of predicted matches. If multiple queries were
            provided, returns the best match for each query.
        """

        pass
    
    @abstractmethod
    def predict_proba(
            self, X: Union[str, List[str]]
        ) -> List[List[Tuple[str, float]]]:
        """
        Predict matches with probability scores for the given queries.
        
        Parameters
        ----------
        X : str or List[str]
            The query or queries to predict matches for.
            
        Returns
        -------
        List[List[Tuple[str, float]]]
            For each query, a list of tuples containing (match, score).
            Scores are typically similarity or distance measures.
        """

        pass

    def _validate_query(self, query: Union[str, List[str]]) -> List[str]:
        """
        Validate the query.
        
        Parameters
        ----------
        query : str or List[str]
            The query to validate.
            
        Returns
        -------
        List[str]
            The validated query as a list.
            
        Raises
        ------
        ValueError
            If query is not a string or list of strings.
        """
        
        # If the query is a string, convert it to a list.
        if isinstance(query, str):
            query = [query]
            
        # Validate the query.
        if not isinstance(query, list):
            raise ValueError(
                "Query must be a string or a list of strings."
            )
            
        return query

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
                "This predictor instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using "
                "this estimator."
            )
