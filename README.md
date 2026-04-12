# AI Diary — Emotion Classification System

Final Year Project | BSc (Hons) Data Science and Analytics | University of Westminster

---

## Project Overview

The AI Diary is an AI-powered reflective diary system that detects fine-grained emotions from user-written text and responds with short, empathetic reflections. The system is built on a fine-tuned RoBERTa-base transformer trained on the GoEmotions dataset across 28 emotion categories.

The project explores how transformer-based NLP models can support emotional self-awareness through free-form diary writing, with an emphasis on ethical, non-clinical, and supportive design. Rather than asking users to select a mood from a list, the AI Diary processes natural, unstructured writing and returns a contextually appropriate reflection based on what the model detects.

---

## Repository Contents

| File | Description |
|---|---|
| `01_EDA.ipynb` | Exploratory data analysis of the GoEmotions dataset including class distribution, text length analysis, and label frequency charts |
| `02_IPD.ipynb` | ClinicalBERT baseline and RoBERTa prototype trained on 5,000 samples for one epoch — submitted as the Interim Progress Demonstration |
| `03_FinalModel.ipynb` | Full RoBERTa training on 43,410 samples across three epochs with class-weighted loss, test set evaluation, survey analysis, and reflection system |
| `app.py` | Streamlit dashboard — run locally with the trained model for the full interactive prototype |
| `requirements.txt` | Python dependencies for local and cloud deployment |
| `survey1_psychology.csv` | Survey 1 responses — 11 psychology student participants |
| `survey2_General.csv` | Survey 2 responses — 20 general participants including open-text diary entries used for model validation |
| `README.md` | This file |

---

## Dataset

The project uses the GoEmotions dataset published by Google Research (Demszky et al., 2020).

- 58,009 Reddit comments annotated across 28 fine-grained emotion categories
- Multi-label — a single comment can carry more than one emotion simultaneously
- Pre-divided into fixed train, development, and test splits

| Split | Samples | Purpose |
|---|---|---|
| Training | 43,410 | Used to train the model |
| Development | 5,426 | Used to monitor performance during training |
| Test | 5,427 | Held back entirely for final evaluation |

The dataset is not included in this repository due to licensing restrictions. Download the official splits from:

https://github.com/google-research/google-research/tree/master/goemotions

---

## Final Model Results

| Metric | ClinicalBERT Baseline | RoBERTa Prototype | Final RoBERTa Model |
|---|---|---|---|
| Macro F1 | 0.0220 | 0.0178 | 0.5151 |
| Micro F1 | 0.3190 | 0.3035 | 0.5269 |
| Weighted F1 | N/A | N/A | 0.4998 |
| Training samples | 5,000 | 5,000 | 43,410 |
| Epochs | 1 | 1 | 3 |
| Class-weighted loss | No | No | Yes |

The jump from Macro F1 0.02 to 0.52 was produced by three compounding changes: using the full training set rather than a 5,000-sample subset, training for three epochs rather than one, and introducing class-weighted binary cross-entropy loss to address the severe label imbalance in the dataset.

---

## Key Technical Decisions

- **Sigmoid activation** instead of softmax — allows multi-label prediction so multiple emotions can be detected from a single entry simultaneously
- **Threshold 0.3** — tuned during the prototype phase to improve recall on rare emotions that the model assigns probabilities between 0.3 and 0.5
- **Class-weighted BCEWithLogitsLoss** — rare emotions like grief receive higher weights so the model is penalised more heavily for missing them
- **RoBERTa-base** — outperforms domain-specific ClinicalBERT on informal conversational text
- **Template-based reflections** — 28 pre-written responses for ethical safety and predictability in a mental health adjacent context

---

## Running the Streamlit App

The app requires the trained model saved locally. Update `MODEL_PATH` in `app.py` to point to your local model folder, then run:

```bash
pip install streamlit torch transformers pandas matplotlib
streamlit run app.py
```

The trained model is not included in this repository due to its size of approximately 500MB. It is saved to Google Drive at the end of Notebook 03 and can be downloaded from there.

If the model folder is not found, the app automatically falls back to keyword-based emotion detection so the interface still works for demonstration purposes without the trained model.

---

## Survey Data

Two surveys were conducted as part of the project validation.

- `survey1_psychology.csv` — 11 psychology student responses covering comfort scores, feature preferences, and ethical considerations around the AI Diary concept
- `survey2_General.csv` — 20 general participant responses including open-text diary entries that were processed through the trained model to produce a real-world agreement score of 0.1471

---

## Reproducibility

All experiments were conducted in Google Colab using Python and HuggingFace Transformers. Notebooks can be executed after downloading and uploading the GoEmotions dataset. The trained model is saved to Google Drive at the end of Notebook 03 and reloaded from there in subsequent sessions.

---

## Video Demonstrations

**FYP Demo Video**
[TO BE ADDED]

**IPD Demo Video**
https://youtu.be/z1PQxvvUhIw

---

## Assessment Context

This repository supports the Final Year Project submission for module 6DATA007W at the University of Westminster. It also covers the earlier Project Proposal Specification and Interim Progress Demonstration.

---

## Author

Faisal Qaderi
W1897498
BSc (Hons) Data Science and Analytics
University of Westminster
Supervisor: Dr Habeeb Bolugan
