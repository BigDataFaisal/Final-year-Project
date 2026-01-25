# AI-Powered Reflective Diary – Emotion Classification

This repository contains the interim implementation for a Final Year Project exploring
fine-grained emotion classification to support an AI-powered reflective diary system.

The project investigates how natural language processing techniques can be used to
identify emotions from short reflective text, with an emphasis on ethical, non-clinical,
and user-supportive design.

---

## Project Objectives
- Explore and analyse the GoEmotions dataset
- Develop a reproducible preprocessing and modelling pipeline
- Implement a baseline multi-label emotion classifier
- Prototype a transformer-based model for emotion detection
- Evaluate model performance using appropriate multi-label metrics
- Establish a foundation for final system development

---

## Dataset
This project uses the **GoEmotions** dataset, which contains approximately 43,000 Reddit
comments annotated across 28 emotion categories.

Due to licensing restrictions, the dataset is **not included** in this repository.

Dataset source:  
https://github.com/google-research/google-research/tree/master/goemotions


## Models Implemented

### Baseline Model
- TF-IDF vectorisation (unigrams and bigrams)
- One-vs-Rest Logistic Regression
- Evaluated using macro and micro F1-score

### Transformer Prototype
- RoBERTa-base
- Multi-label classification with sigmoid activation
- Trained on a reduced subset of the dataset for computational feasibility
- Evaluated using macro and micro F1-score


## Interim Results
- The baseline model provides a useful performance benchmark
- The transformer-based approach shows improved performance for frequently occurring emotions
- Results highlight challenges related to class imbalance and fine-grained emotion detection

Visualisations and performance comparisons are available in the `figures/` directory.


## Reproducibility
All experiments were conducted in **Google Colab** using Python.
Each notebook can be run independently after downloading the GoEmotions dataset
and uploading it to the Colab environment.


## Assessment Context
This repository supports the following assessments:
- **Project Proposal and Specification (PPS)**
- **Interim Progress Demonstration (IPD)**
- **Final Year Project (FYP)**

## Author
Student Name  Faisal Qaderi
BSc Data Science and Analytics  
University of Westminster
