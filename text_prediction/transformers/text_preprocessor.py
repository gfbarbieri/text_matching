"""
Text preprocessing module for text classification tasks.

This module provides a flexible text preprocessing pipeline that can be
used with scikit-learn, allowing custom text processors to be integrated
into prediction pipelines.
"""

########################################################################
## IMPORTS
########################################################################

from typing import List, Callable
from sklearn.base import BaseEstimator, TransformerMixin

########################################################################
## TEXT PREPROCESSOR
########################################################################

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that applies a sequence of
    text preprocessing functions to each document.
    
    This preprocessor can be included in a scikit-learn pipeline and
    will transform text inputs according to the registered preprocessing
    functions.
    
    Parameters
    ----------
    transformers : list of callable, default=None
        A list of functions that take a string as input and return
        a transformed string.
    
    Attributes
    ----------
    transformers : list of callable
        The list of transformer functions to be applied to each text.
    
    Examples
    --------

    .. code-block:: python

        from text_predict.text_preprocessing import TextPreprocessor

        # Define custom preprocessing functions
        def lowercase(text):
            return text.lower()

        def remove_special_chars(text):
            import re
            return re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # Create a preprocessor with these functions 
        preprocessor = TextPreprocessor(
            transformers=[lowercase, remove_special_chars]
        )

        # Apply to text examples
        texts = ["Hello, World!", "This is a Test."]
        preprocessor.transform(texts)
        # ['hello world', 'this is a test']

    """

    def __init__(self, transformers: List[Callable] = None) -> None:
        """
        Initialize the TextPreprocessor with a list of transformer
        functions.
        
        Parameters
        ----------
        transformers : list of callable, default=None
            A list of functions that take a string as input and return a
            transformed string.
        """

        self.transformers = transformers or []
    
    def fit(self, X: List[str], y=None) -> 'TextPreprocessor':
        """
        No-op fit method to conform to scikit-learn's transformer
        interface.
        
        Parameters
        ----------
        X : list of str
            The input documents to fit.
        y : ignored
            Included for consistency with scikit-learn's API.
        
        Returns
        -------
        self : TextPreprocessor
            The fitted transformer.
        """
        
        return self

    def transform(self, X: List[str]) -> List[str]:
        """
        Transform the input text using the sequence of transformers.
        
        Parameters
        ----------
        X : list of str
            The input documents to transform.
            
        Returns
        -------
        list of str
            The transformed documents.
        """
        
        # Initialize an empty list to store the transformed texts.
        transformed = []

        # Loop over each text document and apply each transformer.
        for i, text in enumerate(X):

            transformed_text = text

            # For each transformer, apply it to the text document.
            for j, func in enumerate(self.transformers):

                # Apply the transformer to the text. If the transformer
                # fails, raise an error.
                transformed_text = func(transformed_text)
            
            # Append the transformed text to the list of transformed
            # texts.
            transformed.append(transformed_text)
        
        return transformed