# Spam or Ham: NLP Take-Home Assessment (Track A)

SMS spam classification on the UCI SMS Spam Collection: tokenization comparison, TF-IDF, Logistic Regression and error analysis.

Everything is in **`spam_or_ham.ipynb`**, saved with all outputs.

## Run

**Colab:** upload the notebook and choose *Runtime → Run all*. The dataset is downloaded automatically from the UCI repository.

**Locally:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook spam_or_ham.ipynb
```

## Results (held-out test set, 1,035 messages)

| Accuracy | Spam precision | Spam recall | Spam F1 |
|---|---|---|---|
| 97.8% | 0.91 | 0.92 | 0.91 |

Pipeline: duplicates removed → NLTK `TweetTokenizer` → TF-IDF (uni+bigrams, `min_df=2`, stop words kept) → Logistic Regression (`class_weight='balanced'`). All hyperparameters were chosen with 5-fold cross-validation on the training set only.
