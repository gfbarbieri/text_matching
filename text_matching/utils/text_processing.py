"""
Common text preprocessing functions for text classification tasks.

This module provides a collection of commonly used text preprocessing
functions that can be used with the TextPreprocessor class or
independently.
"""

########################################################################
## IMPORTS
########################################################################

import re
import string
import unicodedata

from typing import Callable

########################################################################
## TEXT CLEANING FUNCTIONS
########################################################################

def lowercase(text: str) -> str:
    """
    Convert text to lowercase.
    
    Parameters
    ----------
    text : str
        Input text.
        
    Returns
    -------
    str
        Lowercase text.
    """

    # Convert the text to lowercase.
    text = text.lower()

    return text

def remove_punctuation(text: str) -> str:
    """
    Remove all punctuation from text.
    
    Parameters
    ----------
    text : str
        Input text.
        
    Returns
    -------
    str
        Text with punctuation removed.
    """

    # Create a translator that maps punctuation to an empty string.
    translator = str.maketrans('', '', string.punctuation)
    
    # Translate the text.
    text = text.translate(translator)

    return text

def remove_numbers(text: str) -> str:
    """
    Remove all numbers from text.
    
    Parameters
    ----------
    text : str
        Input text.
        
    Returns
    -------
    str
        Text with numbers removed.
    """

    # Remove the numbers.
    text = re.sub(r'\d+', '', text)

    return text

def remove_whitespace(text: str) -> str:
    """
    Remove excess whitespace from text.
    
    Parameters
    ----------
    text : str
        Input text.
        
    Returns
    -------
    str
        Text with normalized whitespace.
    """

    # Normalize the whitespace.
    text = ' '.join(text.split())

    return text

def remove_special_chars(text: str, keep: str = '') -> str:
    """
    Remove special characters from text.
    
    Parameters
    ----------
    text : str
        Input text.
    keep : str, default=''
        Characters to keep (in addition to alphanumeric and whitespace).
        
    Returns
    -------
    str
        Text with special characters removed.
    """

    # Define the regex pattern to match special characters.
    pattern = r'[^a-zA-Z0-9\s' + re.escape(keep) + ']'

    # Replace the special characters with an empty string.
    text = re.sub(pattern, '', text)

    return text

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters to their closest ASCII representation.
    
    Parameters
    ----------
    text : str
        Input text.
        
    Returns
    -------
    str
        Text with normalized Unicode characters.
    """

    # Normalize the Unicode characters to their closest ASCII
    # representation.
    normalized = (
        unicodedata.normalize('NFKD', text)
        .encode('ASCII', 'ignore')
        .decode('ASCII')
    )

    return normalized

########################################################################
## FACTORY FUNCTIONS
########################################################################

def create_regex_replacer(
        pattern: str, replacement: str=''
    ) -> Callable[[str], str]:
    """
    Create a function that replaces text matching a regex pattern.
    
    Parameters
    ----------
    pattern : str
        Regular expression pattern to match.
    replacement : str, default=''
        String to replace matches with.
        
    Returns
    -------
    callable
        Function that takes a string and returns it with replacements
        applied.
    """

    # Compile the regex pattern.
    compiled_pattern = re.compile(pattern)
    
    # Define the replacer function.
    def replacer(text: str) -> str:
        return compiled_pattern.sub(replacement, text)
        
    return replacer