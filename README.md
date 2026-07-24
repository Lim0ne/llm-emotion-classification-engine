# Multi-Label Customer Emotion Classification Engine 

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9DC3E?logo=huggingface)](https://huggingface.co/)

## Project Overview
In modern conversational AI and B2B chatbot solutions, understanding nuanced human emotion is critical for intelligent routing (e.g., escalating angry/disgusted customers to human agents). This project develops a robust **multi-label text classification pipeline** capable of simultaneously detecting 11 independent emotional states from short, noisy social media texts.

By progressively iterating from a naive TF-IDF baseline to a fully fine-tuned **Twitter-RoBERTa-base** transformer, this engine successfully surpassed the SemEval 2018 Task 1 State-of-the-Art (SOTA) benchmark (Macro F1: 0.528), achieving a peak **Macro F1 score of 0.5925**.

---

## Data Pipeline & EDA: Handling Severe Class Imbalance
The dataset contains highly imbalanced, non-mutually exclusive labels. 

*   **The Challenge:** The most frequent label (`disgust`, 2,602 samples) outweighs the least frequent (`trust`, 357 samples) by a ratio of **7.3x**. A naive model would simply suppress minority classes to artificially boost standard accuracy.
*   **The Solution:** I engineered a custom **Weighted Binary Cross-Entropy (BCE) Loss** function. By computing a dynamic `pos_weight` ratio for each class (scaling up to 18.15x penalty for errors on minority classes like `trust` and `surprise`), the model was forced to maintain sensitivity across the entire emotional spectrum.

> *[Action Required: Upload your Class Imbalance Bar Chart here e.g., `![Label Distribution & Imbalance](imbalance_ratio.png)`]*

---

## Architecture Evolution & Fine-Tuning Constraints
Building a production-ready model requires understanding the limits of simpler architectures. This project documents a rigorous architectural evolution:

### 1. Baseline: Dense Network + TF-IDF (The Bottleneck)
*   **Result:** Macro F1 0.4878.
*   **Constraint Identified:** The model exhibited severe overfitting from Epoch 1. TF-IDF fundamentally ignores word order and contextual negation (e.g., "I am not happy" vs. "I am happy" yield nearly identical vectors), proving that semantic context is non-negotiable.

### 2. Intermediate Architectures: BiLSTM + GloVe & Attention
*   **Result:** Macro F1 0.5143.
*   **Optimization:** Replaced sparse TF-IDF arrays with pre-trained GloVe embeddings and a Bidirectional LSTM. Introduced a Bahdanau-style additive Attention layer to selectively weight emotionally salient tokens. 
*   **Limitation:** Static embeddings cannot handle polysemy (words having multiple meanings depending on context).

### 3. Production Model: Fine-Tuned Twitter-RoBERTa
**Motivation:** While BERT learned from formal texts (Wikipedia), our business logic operates on noisy social media data. `cardiffnlp/twitter-roberta-base` was pre-trained on 58 million tweets, providing critical domain-specific understanding of informal syntax and sentiment.

**Training Dynamics & Threshold Optimization:**
*   **Early Stopping:** Triggered dynamically based on Validation Macro F1 rather than Val Loss to prioritize class balance.
*   **Threshold Tuning:** Conducted rigorous threshold tuning on the dev set to optimize for the multi-label nature. A threshold of **`0.45`** emerged as the optimal cutoff point, pushing the model to a peak Dev Macro F1 of `0.5997`.

> *[Action Required: Upload your Threshold Tuning Curve or Model Comparison Chart here e.g., `![Threshold Tuning Optimization](roberta_threshold_tuning.png)`]*

---

## Final Results & Business Value
Evaluated on a strictly isolated Test Set (3,259 samples) at the optimized threshold of `0.45`:

| Metric | TF-IDF Baseline | BiLSTM + Attention | **Twitter-RoBERTa** | *SemEval 2018 SOTA* |
| :--- | :--- | :--- | :--- | :--- |
| **Macro F1** | 0.4878 | 0.5143 | **0.5925** | *0.528* |
| **Micro F1** | 0.5694 | 0.5763 | **0.6939** | *0.701* |
| **Hamming Loss**| 0.2090 | 0.2276 | **0.1486** | *-* |

**Actionable Insights:**
*   **Domain Pre-training is Key:** Swapping general BERT for domain-specific RoBERTa yielded consistent gains across the board.
*   **Commercial Viability:** The final engine significantly outperforms the 2018 competition winner in Macro F1 (0.5925 vs 0.528), making it a robust and reliable backend component for dynamic conversational workflows, such as intelligent ticket routing and real-time customer intent tracking.

---

## Repository Structure
*   `01_EDA_and_Preprocessing.ipynb`: Data cleaning, tokenization, and `pos_weight` generation pipeline.
*   `02_RoBERTa_FineTuning_Evaluation.ipynb`: End-to-end model training, validation, threshold tuning (at 0.45), and evaluation using PyTorch and HuggingFace.
