# Hybrid Book Recommendation System

A book recommendation project built on the **Goodbooks-10k** dataset, comparing traditional collaborative filtering with neural two-stage recommender systems.

The project evaluates three main approaches:

- **SVD** collaborative filtering using Surprise
- **Basic TensorFlow Recommenders model** with separate retrieval and ranking stages
- **Hybrid TensorFlow Recommenders model** that incorporates book metadata such as author, publication year, and tags

The goal is not only to predict user ratings accurately, but also to evaluate the quality of generated Top-N recommendations.

---

## Architecture

The TensorFlow Recommenders models use a two-stage recommendation pipeline:

### 1. Retrieval

The retrieval model selects a smaller set of potentially relevant books from a larger candidate set.

The user and book are represented as embedding vectors, and their affinity is calculated using a dot product.

### 2. Ranking

The retrieved books are passed to a ranking model, which predicts how highly the user is expected to rate each book.

The books are then sorted by their predicted ratings to produce the final Top-N recommendations.

---

## Models

### SVD

A matrix-factorization baseline implemented using the Surprise library.

It is used primarily as a traditional collaborative-filtering benchmark.

### Basic TFRS

The basic TensorFlow Recommenders model learns:

- a user ID embedding
- a book ID embedding

Separate user and book models are trained for retrieval and ranking.

### Hybrid TFRS

The hybrid model extends the book representation with additional metadata:

- book ID
- author
- publication year
- top book tags

Categorical features such as book IDs, authors, and tags are represented using learned embeddings.

Publication year is treated as a numerical feature and normalized before being combined with the other book features.

The tag embeddings of each book are aggregated into a single representation and combined with the other book features to create the final book embedding.

---

## Evaluation

The models are evaluated using several groups of metrics.

### Rating prediction

- RMSE
- MAE

### Top-N recommendation quality

- Precision
- Recall
- NDCG
- Hit Rate
- Cumulative Hit Rate
- Rating Hit Rate

### Beyond-accuracy metrics

- Catalog Coverage
- Diversity
- Novelty

Top-N metrics are calculated using sampled candidate sets containing relevant items and sampled unseen books.

---

## Results

| Metric | SVD | Basic TFRS | Hybrid TFRS without tags | Hybrid TFRS |
|---|---:|---:|---:|---:|
| RMSE | **0.8337** | 0.8559 | 0.8551 | 0.8545 |
| MAE | **0.6442** | 0.6763 | 0.6737 | 0.6729 |
| Precision | 0.2855 | 0.4469 | 0.4425 | **0.4556** |
| Recall | 0.1564 | 0.2412 | 0.2385 | **0.2454** |
| NDCG | 0.3029 | 0.4639 | 0.4575 | **0.4706** |
| Hit Rate | 0.1775 | 0.2471 | 0.2423 | **0.2478** |
| Cumulative Hit Rate | 0.2227 | 0.2915 | 0.2855 | **0.2943** |
| Catalog Coverage | **0.7671** | 0.5915 | 0.6120 | 0.6070 |
| Diversity | **0.3831** | 0.3309 | 0.2943 | 0.2970 |
| Novelty | **14.0983** | 14.0277 | 13.8706 | 13.8841 |

---

## Key Findings

The experiments show a trade-off between rating prediction accuracy and Top-N recommendation quality.

SVD achieves the lowest RMSE and MAE, making it the strongest model for explicit rating prediction.

However, the TensorFlow Recommenders models perform substantially better on ranking-based recommendation metrics such as Precision, Recall, NDCG, and Hit Rate.

Adding only author and publication year metadata did not improve Top-N performance compared with the basic TFRS model.

After adding book tags, the hybrid model achieved the best Precision, Recall, NDCG, Hit Rate, and Cumulative Hit Rate among the tested models.

This suggests that richer content-based metadata, particularly book tags, provides useful information for recommendation beyond the collaborative signal learned from user and book IDs.

---

## Dataset

The project uses the **Goodbooks-10k** dataset.

The dataset contains:

- user ratings
- book metadata
- authors
- publication years
- Goodreads tags
- book-tag interactions

The dataset itself is not included in this repository.

To run the project, place the Goodbooks-10k files inside:

```text
goodbooks-10k-master/
```

The code expects files including:

```text
ratings.csv
books.csv
book_tags.csv
tags.csv
```

---

## Project Structure

```text
DataLoader.py
    Loads ratings, book metadata, and tags.

EvaluationData.py
    Creates train/test splits, leave-one-out data, candidate sets,
    popularity statistics, and similarity data.

EvaluationMetrics.py
    Implements rating, ranking, and beyond-accuracy metrics.

AlgorithmEvaluation.py
    Runs the evaluation pipeline for each recommender.

Evaluator.py
    Coordinates model evaluation and result reporting.

TFRSModels.py
    Defines the retrieval, ranking, and hybrid book models.

TFRSBuilders.py
    Contains shared TFRS model-building and training functions.

BasicTFRS.py
    Defines the basic ID-based TFRS recommender.

HybridTFRS.py
    Defines the metadata-enhanced hybrid TFRS recommender.

TFRSAlgorithm.py
    Provides a common interface for training, testing,
    and generating recommendations with TFRS models.

Fight.py
    Main entry point for running and comparing the models.
```

---

## Technologies

- Python
- TensorFlow
- TensorFlow Recommenders
- Keras / tf-keras
- Surprise
- pandas
- NumPy

---

## Running the Project

Create a Python environment with the required dependencies and place the Goodbooks-10k dataset in the expected directory.

Then run:

```bash
python Fight.py
```

The script trains the selected recommenders and prints the evaluation results.

The TensorFlow models were developed and evaluated on CPU.

---

## Future Work

Possible extensions include:

- hyperparameter tuning
- experimenting with different embedding dimensions
- tuning learning rates and number of training epochs
- comparing different numbers of book tags
- improving tag aggregation
- evaluating additional candidate-set sizes
- testing additional recommendation architectures
