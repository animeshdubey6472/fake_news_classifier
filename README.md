# Fake News Classification (BoW → TF-IDF → PAC → LSTM)

## Overview
This project analyzes fake news classification by varying:
- the input text (titles vs full articles),
- the text representation (Bag of Words, TF-IDF, word embeddings),
- and the model (Naive Bayes, Passive Aggressive Classifier, LSTM).

The focus is on how accuracy changed across approaches and why.

---

## 1️⃣ Title-based Classification  
**(Bag of Words only)**

Only news titles were used in this stage. Since titles are short and keyword-focused, Bag of Words (BoW) representation was applied.

| Model | Representation | Accuracy |
|-----|---------------|---------|
| Naive Bayes | BoW | 93.8% |
| Naive Bayes (tuned) | BoW | 95.4% |
| Passive Aggressive Classifier | BoW | 94.8% |

### Observations
- Titles contain strong keywords but limited context
- BoW works well, but performance is capped by short text length
- PAC remained strong, but gains were limited due to less information

---

## 2️⃣ Full Text Classification  
**(TF-IDF features)**

In this stage, full article text was used. Due to longer and noisier content, TF-IDF was used instead of BoW.

| Model | Representation | Accuracy |
|-----|---------------|---------|
| Naive Bayes | TF-IDF | 93.3% |
| Naive Bayes (tuned) | TF-IDF | 95.0% |
| Passive Aggressive Classifier | TF-IDF | 99.5% |

### Observations
- Full articles provide much richer information
- TF-IDF reduces the impact of frequent but uninformative words
- PAC benefited the most due to its margin-based learning and robustness to sparse features

---

## 3️⃣ Deep Learning Approach  
**(Word Embeddings + LSTM on full text)**

A deep learning model was trained using word embeddings instead of sparse text vectors.

| Model | Representation | Accuracy |
|-----|---------------|---------|
| LSTM | Tokenizer + Word Embeddings | ~94.8% |

### Observations
- LSTM captures word order and contextual information
- For this dataset, keyword presence mattered more than long-range dependencies
- Sparse linear models generalized better than deep models

---

## Key Takeaways
- Titles with Bag of Words gave strong but limited performance
- Full text with TF-IDF enabled significantly higher accuracy
- Passive Aggressive Classifier achieved the best overall accuracy (99.5%)
- Feature representation had a greater impact than model complexity

---

## Libraries Used
Python, NumPy, Pandas, NLTK, Scikit-learn, TensorFlow/Keras, Matplotlib

---

## Notes
- Dataset and trained models are intentionally excluded
- The project focuses on accuracy progression and method comparison
