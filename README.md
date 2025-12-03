#  Citation Prediction with LightGBM

A machine learning project that predicts the number of citations an academic paper will receive based on metadata and textual features. The model leverages **LightGBM** with **Optuna** hyperparameter optimization and extensive NLP-based feature engineering.

---

##  Overview

This project builds a regression model to predict citation counts for academic papers using:
- **Text Processing**: Abstract cleaning, tokenization, lemmatization
- **Topic Modeling**: Latent Dirichlet Allocation (LDA) for semantic features
- **Feature Engineering**: Readability scores, temporal features, interaction terms
- **Hyperparameter Tuning**: Bayesian optimization with Optuna
- **Model**: LightGBM Regressor with L1 loss objective

---

##  Project Structure

```
ML_Project/
├── lightgbm3.py        # Main training script
├── train.json          # Training data (required)
├── test.json           # Test data (required)
├── predicted.json      # Output predictions
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

##  Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/citation-prediction.git
   cd citation-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (handled automatically on first run)
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   ```

---

##  Data Format

The model expects JSON files with the following structure:

### Input Schema

| Field        | Type     | Description                              |
|--------------|----------|------------------------------------------|
| `abstract`   | string   | Paper abstract text                      |
| `title`      | string   | Paper title                              |
| `authors`    | string   | Comma-separated author names             |
| `venue`      | string   | Publication venue/journal                |
| `year`       | int      | Publication year                         |
| `references` | list     | List of reference paper IDs              |
| `n_citation` | int      | Number of citations (target, train only) |

### Example

```json
[
  {
    "abstract": "We present a novel approach to...",
    "title": "Deep Learning for Citation Analysis",
    "authors": "John Doe, Jane Smith",
    "venue": "NeurIPS",
    "year": 2020,
    "references": ["paper_id_1", "paper_id_2"],
    "n_citation": 45
  }
]
```

---

## Usage

1. **Prepare your data**
   - Place `train.json` and `test.json` in the appropriate directory
   - Update file paths in `lightgbm3.py` if needed

2. **Run the training script**
   ```bash
   python lightgbm3.py
   ```

3. **Output**
   - Predictions are saved to `predicted.json`
   - Training/validation MAE scores are logged to console

---

##  Features Engineering

### Text Processing Pipeline
1. **Cleaning**: Remove URLs, punctuation, and special characters
2. **Tokenization**: Split text into word tokens
3. **Stopword Removal**: Filter common English stopwords
4. **Lemmatization**: Reduce words to base forms with POS tagging

### Engineered Features

| Feature | Description |
|---------|-------------|
| `years_since_pub` | Time since publication (2024 - year) |
| `avg_citations_year` | Mean citations for papers in same year |
| `num_authors` | Number of authors on the paper |
| `num_references` | Number of cited references |
| `lemmatized_abstract_length` | Length of processed abstract |
| `flesch_readability_score` | Flesch reading ease score |
| `title_length` | Character count of title |
| `venue_encoded` | Target-encoded venue (mean citations) |
| `colaboration_trend` | Interaction: years × num_authors |
| `readability_length_interaction` | Interaction: readability × abstract_length |
| `topic_0` to `topic_9` | LDA topic probabilities (10 topics) |

### TF-IDF Features
- **Authors**: Unigrams and bigrams (max 2,000 features)
- **Title**: Up to trigrams (max 2,000 features)
- **Abstract**: Up to trigrams (max 3,000 features)

---

##  Model Configuration

### LightGBM Hyperparameters (Optimized via Optuna)

| Parameter | Search Range |
|-----------|--------------|
| `n_estimators` | 100 - 2,000 |
| `learning_rate` | 0.01 - 0.3 (log scale) |
| `max_depth` | 3 - 25 |
| `num_leaves` | 15 - 255 |
| `subsample` | 0.6 - 1.0 |
| `colsample_bytree` | 0.6 - 1.0 |
| `reg_alpha` | 0.0 - 1.0 |
| `reg_lambda` | 0.0 - 1.0 |
| `min_data_in_leaf` | 10 - 100 |
| `max_bin` | 63 - 511 |

### Fixed Settings
- **Objective**: `regression_l1` (MAE)
- **Metric**: Mean Absolute Error
- **Optuna Trials**: 30
- **Target Transform**: `log1p` / `expm1`

---

##  Evaluation

The model is evaluated using **Mean Absolute Error (MAE)** on:
- Training set
- Validation set (1/3 split from training data)

A baseline **DummyRegressor** is included for comparison.

---

##  Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
optuna>=3.0.0
nltk>=3.6.0
```

---

##  Notes

- The model uses **log transformation** on the target variable to handle skewed citation distributions
- **Target encoding** for venues helps capture venue prestige
- **LDA topic modeling** extracts latent semantic themes from abstracts
- Optuna uses **TPE sampler** with 3 parallel jobs for efficient search




---

##  Acknowledgments

- [LightGBM](https://github.com/microsoft/LightGBM) - Gradient boosting framework
- [Optuna](https://optuna.org/) - Hyperparameter optimization
- [NLTK](https://www.nltk.org/) - Natural language processing toolkit

