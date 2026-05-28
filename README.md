#  Sentiment Analysis

> Binary sentiment classifier (Positive/Negative) built on IMDB reviews. Covers the full NLP pipeline — text cleaning, TF-IDF + Logistic Regression baseline, and DistilBERT inference — with saved models, evaluation metrics, confusion matrices, and a standalone predict script. Python · scikit-learn · HuggingFace · NLTK

---

##  Project Overview

This project implements an **end-to-end sentiment analysis pipeline** that classifies text (movie reviews, tweets, comments) as **Positive** or **Negative**. It covers the full ML workflow — from raw data loading and text cleaning through model training, evaluation, and reusable inference.

Two approaches are implemented and compared side-by-side:

| Approach | Model | Expected Accuracy |
|---|---|---|
| Baseline | TF-IDF + Logistic Regression | ~89% |
| Advanced | DistilBERT (HuggingFace, pre-trained) | ~93% |

---

##  Project Structure

```
sentiment-analysis/
│
├── sentiment_analysis_task3.ipynb   # Main training notebook (all 8 sections)
├── predict.py                       # Standalone inference script
├── README.md                        # This file
│
├── guide.html                       #  Complete interactive project guide (see below)
│
├── lr_sentiment_model.pkl           # Saved Logistic Regression model (generated)
├── tfidf_vectorizer.pkl             # Saved TF-IDF vectorizer (generated)
│
└── outputs/                         # Generated after running the notebook
    ├── confusion_matrix_baseline.png
    ├── confusion_matrix_bert.png
    ├── model_comparison.png
    ├── top_features.png
    ├── class_distribution.png
    └── metrics_table.csv
```

---

##  Complete Interactive Guide (HTML)

A self-contained **interactive HTML guide** is included — open it in any browser for a full project walkthrough with tabbed navigation:

```
guide.html
```

**What the guide covers:**

| Tab | Contents |
|---|---|
| **Datasets** | All 4 recommended datasets with download links and code snippets |
| **Steps & Code** | 6-step pipeline from setup → cleaning → training → evaluation |
| **Deliverables** | Interactive checklist of all submission items |
| **Rubric** | Scoring breakdown with tips to maximise marks |

**To open:** double-click `guide.html` in your file explorer, or run:
```bash
open guide.html          # macOS
start guide.html         # Windows
xdg-open guide.html      # Linux
```

> No internet connection required the guide is fully self-contained.

---

##  Dataset

**Primary: IMDB Movie Reviews**

- **50,000** labelled movie reviews (25k train / 25k test)
- Binary labels: `1 = Positive`, `0 = Negative`
- Perfectly balanced classes
- Loads automatically via HuggingFace — no manual download needed

```python
from datasets import load_dataset
ds = load_dataset("imdb")
```

**Alternatives supported in the notebook:**

| Dataset | Size | Source | Use case |
|---|---|---|---|
| Sentiment140 | 1.6M tweets | Kaggle / Stanford | Social media / Twitter |
| SST-2 | ~11k sentences | HuggingFace | BERT fine-tuning |
| Reddit Comments | Custom | PRAW API | Live data collection |

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8+
- pip

### Install dependencies

```bash
pip install pandas scikit-learn nltk transformers datasets torch matplotlib seaborn joblib
```

### Download NLTK data

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

> **Note:** The notebook handles both of these automatically when run top-to-bottom.

---

## 🚀 How to Run

### Option 1 — Jupyter Notebook (recommended)

```bash
jupyter notebook sentiment_analysis_task3.ipynb
```

Run all cells from top to bottom. The notebook will:
1. Download the IMDB dataset automatically
2. Clean and vectorise the text
3. Train the TF-IDF + Logistic Regression baseline
4. Run DistilBERT inference on 2,000 test samples
5. Generate and save all plots and metrics

**Expected total runtime:** ~10–20 minutes on CPU (DistilBERT inference is the slowest step)

---

### Option 2 — Standalone Inference Script

After running the notebook once (to generate the saved `.pkl` files):

```bash
# Single review
python predict.py "This movie was absolutely brilliant — a masterpiece!"

# Output:
# Text       : This movie was absolutely brilliant — a masterpiece!
# Sentiment  : Positive
# Confidence : 0.9871
# Prob Neg   : 0.0129
# Prob Pos   : 0.9871
```

```bash
# Run demo examples (no arguments)
python predict.py
```

---

##  Text Cleaning Pipeline

Raw reviews contain HTML tags, URLs, and noise. The cleaning function applied at each step:

```python
def clean_text(text):
    text = text.lower()                          # 1. Lowercase
    text = re.sub(r'<.*?>', ' ', text)           # 2. Strip HTML tags
    text = re.sub(r'http\S+', '', text)          # 3. Remove URLs
    text = re.sub(r"[^a-z\s']", '', text)       # 4. Keep letters only
    tokens = text.split()
    tokens = [w for w in tokens               
              if w not in STOP_WORDS             # 5. Remove stopwords
              and len(w) > 1]                    # 6. Drop single chars
    return ' '.join(tokens)
```

---

##  Results

### Metrics Table

| Model | Accuracy | Precision | Recall | F1 Score | Eval Samples |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | ~0.892 | ~0.891 | ~0.893 | ~0.892 | 25,000 |
| DistilBERT (pre-trained) | ~0.934 | ~0.935 | ~0.933 | ~0.934 | 2,000 |

> Exact values will appear in `outputs/metrics_table.csv` after running the notebook.

### Generated Plots

| File | Description |
|---|---|
| `class_distribution.png` | Train/test class balance bar chart |
| `confusion_matrix_baseline.png` | TF-IDF + LR confusion matrix |
| `confusion_matrix_bert.png` | DistilBERT confusion matrix |
| `model_comparison.png` | Side-by-side accuracy/F1/precision/recall |
| `top_features.png` | Most predictive positive and negative words |

---

##  Notebook Structure (8 Sections)

| Section | Description |
|---|---|
| 1 | Install & import all libraries |
| 2 | Load IMDB dataset + explore class distribution |
| 3 | Text cleaning & preprocessing pipeline |
| 4 | Baseline: TF-IDF vectorisation + Logistic Regression training + evaluation |
| 5 | Advanced: DistilBERT zero-shot inference + evaluation |
| 6 | Side-by-side model comparison with charts |
| 7 | Save trained models to disk (`.pkl`) |
| 8 | Standalone inference function demo |

---

##  Evaluation Rubric

| Criterion | Weight | How this project addresses it |
|---|---|---|
| **Pipeline** | 35% | End-to-end: load → clean → vectorise → train → infer → save |
| **Performance** | 35% | Both models evaluated; metrics table + confusion matrices |
| **Clarity** | 20% | Markdown cells explain every section; plots are labelled |
| **Reusability** | 10% | Models saved to `.pkl`; `predict.py` works standalone |

---

##  Deliverables Checklist

- [x] Training notebook — `sentiment_analysis_task3.ipynb`
- [x] Saved model — `lr_sentiment_model.pkl` + `tfidf_vectorizer.pkl`
- [x] Inference script — `predict.py`
- [x] Metrics table — `outputs/metrics_table.csv`
- [x] Confusion matrix (baseline) — `outputs/confusion_matrix_baseline.png`
- [x] Confusion matrix (BERT) — `outputs/confusion_matrix_bert.png`
- [x] Model comparison chart — `outputs/model_comparison.png`
- [x] Feature importance chart — `outputs/top_features.png`
- [x] Interactive project guide — `guide.html`

---

##  Key Learning Outcomes

- **Text cleaning:** HTML stripping, stopword removal, tokenisation
- **TF-IDF:** Converting raw text to weighted numerical vectors
- **Logistic Regression:** Fast, interpretable baseline classifier
- **Transfer learning:** Using a pre-trained BERT model without fine-tuning
- **Evaluation metrics:** Accuracy, Precision, Recall, F1, Confusion Matrix
- **Model persistence:** Saving and loading sklearn models with `joblib`

---

##  References & Resources

- [HuggingFace IMDB Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)
- [Sentiment140 on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)
- [DistilBERT Model Card](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
- [scikit-learn TF-IDF Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [HuggingFace Transformers — Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
- [GitHub — FinGPT](https://github.com/AI4Finance-Foundation/FinGPT)

---

##  Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
