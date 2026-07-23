# Smart MCQ Solver Challenge

**Deep Learning Project | Term-2 2026 | Student-id: 23f1002033 | Name: Ishank Gupta**
**Live Demo:**  
https://huggingface.co/spaces/IshankGupta/smart-mcq-solver
---

## Overview

Build an AI system capable of solving challenging multiple-choice questions by predicting the **top three most probable answers** from five candidate options (A-E). Unlike standard classification tasks, this competition evaluates a model's ability to **rank answers**, rewarding predictions that place the correct option higher in the ranked list using the **Mean Average Precision at 3 (MAP@3)** metric.

This project explores a progression of approaches from traditional machine learning and deep learning to retrieval-augmented generation (RAG) and ensemble methods to improve reasoning and answer ranking across diverse MCQ domains.

---

## Models

| Model | Type | Score MAP@3 |
| :--- | :--- | ---: |
| TF-IDF + Logistic Regression | Traditional ML | **0.73940** |
| BiLSTM | Deep Learning | **0.75050 (Best Validation)** |
| FAISS + Retrieval-Augmented Generation (RAG) | Retrieval + LLM | **0.74064** |
| Ensemble | Combined | **0.74896 (Final Submission)** |

---

## Dataset

- **Task:** Predict the **top 3 ranked answers** for challenging multiple-choice questions.
- **Input:**
  - Question prompt
  - Five answer choices (A-E)
- **Output:**
  - Three ranked answer labels
- **Evaluation Metric:**
  - Mean Average Precision @3 (**MAP@3**)
- **Challenge Focus:**
  - Natural language understanding
  - Context-aware reasoning
  - Answer ranking
  - Robust generalization across diverse question types

---

## Notebook Structure

```text
0. Setup

1. Evaluation Metric
   - MAP@3 implementation
   - Submission format

2. Exploratory Data Analysis
   - Dataset statistics
   - Question length distribution
   - Answer option analysis
   - Class distribution

3. Shared Utilities
   - Text preprocessing
   - Tokenization
   - Data loaders
   - Helper functions

4. Model 1 - TF-IDF + Logistic Regression

5. Model 2 - BiLSTM

6. Model 3 - Retrieval-Augmented Generation (RAG)

7. Model Comparison

8. Final Inference - Ensemble

9. Submission Generation
```

---

## Requirements

```text
transformers
torch
sentence-transformers
faiss-cpu
scikit-learn
datasets
pandas
numpy
tqdm
wandb
```

---

## Training

All experiments were tracked and compared using **Weights & Biases** under the project:

**23f1002033-t22026**

The training pipeline logs model performance, validation MAP@3, training loss, and experiment configurations, enabling reproducible comparison across different approaches.

---

## Approach

The project progressively develops stronger answer-ranking models:

### Model 1 - TF-IDF + Logistic Regression

A lightweight baseline using sparse TF-IDF representations with Logistic Regression for efficient answer prediction.

### Model 2 - BiLSTM

A bidirectional LSTM model capable of capturing sequential dependencies and semantic relationships between questions and answer choices. This model achieved the **highest MAP@3**.

### Model 3 - Retrieval-Augmented Generation (RAG)

A retrieval-based pipeline that leverages **FAISS** for semantic search and retrieves relevant contextual information before ranking candidate answers, improving reasoning on knowledge-intensive questions.

### Ensemble

The final submission combines predictions from multiple models to leverage their complementary strengths and produce more robust ranked predictions for the competition.

---

## Evaluation

Submissions are evaluated using **Mean Average Precision @3 (MAP@3)**.

Models receive higher scores when the correct answer appears earlier in the ranked predictions.

