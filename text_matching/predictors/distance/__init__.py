"""
This module contains the constants for the fuzzy search.

The constants are used to validate the algorithm and method arguments
in the FuzzySearch class.

The supported algorithms are:
    - "damerau_levenshtein"
    - "hamming"
    - "indel"
    - "jaro"
    - "jaro_winkler"
    - "levenshtein"

The supported methods are:
    - "distance"
    - "normalized_distance"
    - "similarity"
    - "normalized_similarity"

"""

########################################################################
## IMPORTS
########################################################################

from rapidfuzz.distance import (
    DamerauLevenshtein, Hamming, Indel, Jaro, JaroWinkler, Levenshtein
)

########################################################################
## CONSTANTS
########################################################################

ALGORITHMS = {
    "damerau_levenshtein": DamerauLevenshtein,
    "hamming": Hamming,
    "indel": Indel,
    "jaro": Jaro,
    "jaro_winkler": JaroWinkler,
    "levenshtein": Levenshtein
}

METHODS = [
    "distance",
    "normalized_distance",
    "similarity",
    "normalized_similarity"
]