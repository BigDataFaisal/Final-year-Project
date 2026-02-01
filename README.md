# AI-Powered Reflective Diary – Emotion Classification (IPD)

This repository contains the **interim (IPD) implementation** for a Final Year Project investigating **multi-label emotion classification** to support an AI-powered reflective diary system.

The project explores how **transformer-based NLP models** can detect fine-grained emotions from short reflective text, with an emphasis on **ethical, non-clinical, and supportive design**.

---

## Project Objectives
- Analyse and understand the GoEmotions dataset  
- Build a reproducible preprocessing and modelling pipeline  
- Implement a domain-specific transformer baseline  
- Prototype a general-purpose transformer model  
- Evaluate performance using multi-label metrics (macro & micro F1)

---

## Dataset
The project uses the **GoEmotions dataset** (~43,000 Reddit comments, 28 emotion labels).

The dataset is **not included** due to licensing restrictions.

**Dataset source:**  
https://github.com/google-research/google-research/tree/master/goemotions

---

## Models Implemented

### Transformer Baseline
- **ClinicalBERT** (domain-specific transformer)
- Multi-label classification with sigmoid activation
- Trained on a reduced **5,000-sample subset** (IPD feasibility)
- Evaluated using **macro and micro F1-score**

### Transformer Prototype
- **RoBERTa-base**
- Identical experimental setup to the baseline
- Used to assess **domain-specific vs general-purpose** performance

---

## Interim Results
- Both models demonstrate stronger **micro F1** than **macro F1**, reflecting class imbalance
- RoBERTa shows marginally stronger performance on frequent emotions
- Results validate feasibility and highlight areas for final optimisation

Figures and performance comparisons are provided in the repository.

---

## Reproducibility
All experiments were conducted in **Google Colab** using **Python** and **Hugging Face**.  
Notebooks can be executed after downloading and uploading the GoEmotions dataset.

---

## Assessment Context
This repository supports:
- Project Proposal Specification (PPS)
- Interim Progress Demonstration (IPD)
- Final Year Project (FYP)

---

## Author
**Faisal Qaderi**  
BSc (Hons) Data Science and Analytics  
University of Westminster
