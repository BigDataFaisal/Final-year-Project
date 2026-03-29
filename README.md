# AI Diary : Emotion Classification System
Final Year Project | BSc (Hons) Data Science and Analytics | University of Westminster

---

## Project Overview

The AI Diary is an AI-powered reflective diary system that detects fine grained emotions from user-written text and responds with short, empathetic reflections. The system is built on a fine-tuned RoBERTa-base transformer trained on the GoEmotions dataset across 28 emotion categories.

The project explores how transformer based NLP models can support emotional self-awareness through free-form diary writing, with an emphasis on ethical, non-clinical, and supportive design.

---

## Repository Contents

| File / Folder | Description |
|---|---|
| `01_EDA.ipynb` | Exploratory data analysis of GoEmotions dataset |
| `02_IPD.ipynb` | ClinicalBERT baseline and RoBERTa prototype (IPD submission) |
| `03_FinalModel_.ipynb` | Full RoBERTa training, evaluation, survey analysis, reflection system |
| `app.py` | Streamlit dashboard : run locally with the trained model |
| `requirements.txt` | Python dependencies for Streamlit Cloud deployment |
| `survey1.csv` | Survey 1 responses : psychology students |
| `Survey2.csv` | Survey 2 responses : general participants |
| `README.md` | This file |

---

## Dataset

The project uses the GoEmotions dataset published by Google Research.

- 58,009 Reddit comments annotated across 28 emotion categories
- Multi-label — a single comment can carry more than one emotion
- Pre-divided into train (43,410), dev (5,426), and test (5,427) splits

The dataset is not included in this repository due to licensing restrictions. Download the official splits from:

https://github.com/google-research/google-research/tree/master/goemotions

---

## Final Model Results

| Metric | ClinicalBERT Baseline | RoBERTa Prototype | Final RoBERTa Model |
|---|---|---|---|
| Macro F1 | 0.0220 | 0.0178 | 0.5151 |
| Micro F1 | 0.3190 | 0.3035 | 0.5269 |
| Training samples | 5,000 | 5,000 | 43,410 |
| Epochs | 1 | 1 | 3 |
| Class-weighted loss | No | No | Yes |

---

## Key Technical Decisions

- **Sigmoid activation** instead of softmax : allows multi-label prediction
- **Threshold 0.3** : tuned to improve recall on rare emotions
- **Class-weighted BCEWithLogitsLoss** : addresses severe label imbalance
- **RoBERTa-base** : outperforms domain-specific ClinicalBERT on informal text
- **Template-based reflections** : 28 pre-written responses for ethical safety

---

## Running the Streamlit App

The app requires the trained model saved locally. Update MODEL_PATH in app.py to point to your local model folder, then run:
```bash
pip install streamlit torch transformers pandas matplotlib
streamlit run app.py
```

The trained model is not included in this repository due to its size (~500MB). It is saved to Google Drive during Notebook 03 and can be downloaded from there.

If the model folder is not found, the app automatically falls back to keyword-based emotion detection so the interface still works for demonstration purposes.

---

## Reproducibility

All experiments were conducted in Google Colab using Python and HuggingFace Transformers. Notebooks can be executed after downloading and uploading the GoEmotions dataset. The model is saved to Google Drive at the end of Notebook 03.

---

## Survey Data

Two surveys were conducted as part of the project validation.

- `survey1.csv` : 11 psychology student responses on comfort, features, and ethics
- `survey2.csv` : 20 general participant responses including open-text diary entries used for real-world model validation

---

## Video Demo

https://youtu.be/z1PQxvvUhIw

---

## Assessment Context

This repository supports the Final Year Project (FYP) submission for module 6DATA007W at the University of Westminster. It also covers the earlier Project Proposal Specification (PPS) and Interim Progress Demonstration (IPD).

---

## Author

Faisal Qaderi
BSc (Hons) Data Science and Analytics
University of Westminster
Supervisor: Dr Habeeb Bolugan
