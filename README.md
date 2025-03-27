# Text Prediction Library

A Python library for text matching and prediction, providing flexible tools for both supervised and unsupervised text matching tasks. The library supports various text matching strategies including bag-of-words approaches and fuzzy string matching.

## Features

### Text Matching Predictors

- **BOWPredictor**: Bag-of-words based text matching
  - Supports various text vectorization methods (CountVectorizer, TF-IDF, etc.)
  - Configurable n-gram ranges and analyzers
  - Customizable similarity metrics
  - Compatible with scikit-learn's API

- **FuzzyPredictor**: Fuzzy string matching using RapidFuzz
  - Multiple string similarity algorithms (Levenshtein, Jaro, etc.)
  - Configurable similarity/distance methods
  - Support for custom text preprocessing
  - Parallel processing capabilities

### Text Preprocessing

- **TextPreprocessor**: Flexible text preprocessing pipeline
  - Scikit-learn compatible transformer
  - Support for custom preprocessing functions
  - Chain multiple preprocessing steps
  - Easy integration with prediction pipelines

## Installing Dependencies

Poetry:
```bash
poetry install
```

Pip with requirements.txt:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create requirements.txt file with poetry, if required.
```bash
poetry export -f requirements.txt --output requirements.txt
```

## Usage

### Supervised Text Matching

```python
from text_prediction.predictors.vectorized import BOWPredictor
from text_prediction.predictors.distance import FuzzyPredictor

# Initialize predictors.
bow_predictor = BOWPredictor(analyzer='char_wb', ngram_range=(2,3))
fuzzy_predictor = FuzzyPredictor(algorithm="levenshtein", method="similarity")

# Fit on training data.
bow_predictor.fit(X, y)  # y contains reference labels.
fuzzy_predictor.fit(X, y)

# Make predictions.
bow_matches = bow_predictor.predict(new_text)
fuzzy_matches = fuzzy_predictor.predict(new_text)
```

### Unsupervised Text Matching

```python
# Fit on corpus.
bow_predictor.fit(documents)  # documents to match within.
fuzzy_predictor.fit(documents)

# Find similar documents.
similar_docs = bow_predictor.predict(new_documents)
similar_docs = fuzzy_predictor.predict(new_documents)
```

### Text Preprocessing

```python
from text_prediction.transformers import TextPreprocessor

# Define preprocessing functions.
def lowercase(text):
    return text.lower()

def remove_special_chars(text):
    import re
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

# Create preprocessor.
preprocessor = TextPreprocessor(
    transformers=[lowercase, remove_special_chars]
)

# Transform text.
processed_texts = preprocessor.transform(texts)
```

## Features in Detail

### BOWPredictor

- **Vectorization Options**:
  - Custom text vectorizers
  - Configurable analyzers (word, char, char_wb)
  - Adjustable n-gram ranges
  - Support for custom similarity metrics

- **Matching Modes**:
  - Supervised: Match to known labels
  - Unsupervised: Find similar documents
  - Configurable number of matches

### FuzzyPredictor

- **Algorithms**:
  - Levenshtein distance
  - Damerau-Levenshtein distance
  - Hamming distance
  - Jaro similarity
  - Jaro-Winkler similarity
  - Indel distance

- **Features**:
  - Custom text preprocessing
  - Configurable similarity thresholds
  - Parallel processing support
  - Score normalization options

### TextPreprocessor

- **Features**:
  - Chain multiple preprocessing steps
  - Custom preprocessing functions
  - Scikit-learn compatibility
  - Easy pipeline integration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.