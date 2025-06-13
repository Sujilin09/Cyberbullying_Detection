#CYBERBULLYING DETECTION USING A CUSTOM DECEN-INSPIRED DEEP LEARNING APPROACH
==============================================================================

This repository presents a deep learning pipeline for cyberbullying detection,
inspired by the DECEN framework. The model is trained on the Wikipedia Talk 
Labels: Toxicity Dataset, which contains a highly imbalanced class distribution.

DATASET
-------
Source: Wikipedia Talk Labels: Toxicity Dataset

Class Distribution:
- Malicious (1): 10%
- Non-Malicious (0): 90%

DATA PREPARATION STEPS
-----------------------
1. Load and Merge:
   - Loaded train.csv, test.csv, and test_labels.csv.
   - Merged test and test_labels to isolate labeled/unlabeled samples.

2. Text Cleaning:
   - Lowercasing
   - Expanding contractions
   - Removing stopwords and special characters
   - Optionally preserving sentence punctuation

3. Noise Reduction:
   - Removed URLs
   - Replaced repeated characters (e.g., "soooo" -> "so")

4. Export:
   - Saved cleaned datasets for training and evaluation.

BASELINE MODELS AND RESULTS
---------------------------
Model     | Precision(1) | Recall(1) | F1-Score(1) | Avg Precision | ROC AUC
--------- | -------------| ----------| ------------| --------------|---------
BiLSTM    |     0.75     |   0.75    |    0.75     |     0.81      |    -
LSTM      |     0.73     |   0.78    |    0.75     |     0.82      |    -
GRU       |     0.74     |   0.77    |    0.75     |       -       |    -
BGRU      |     0.76     |   0.75    |    0.76     |     0.826     |  0.960
CNN       |     0.81     |   0.66    |    0.73     |       -       |    -

CUSTOM MODEL (DECEN-INSPIRED ARCHITECTURE)
------------------------------------------
Built on the DECEN structure with 3 major modules:

1. DER MODULE (Detection-Emotion Recognition)
   - BiLSTM-CRF for token-level emotion labeling
   - Tasks: NER, POS tagging, Chunking

2. ECR MODULE (Emotion-Context Representation)
   - Inputs: 
     * Emotion label embeddings (from DER)
     * Post embeddings (BERT)
   - Fusion Strategies:
     * Concatenation
     * Standardization of emotion vectors (Improvement 1)
     * Residual Dot Product Fusion (Improvement 2)

3. DETECTION MODULE
   - Initial: BiLSTM-FFNN
   - Final: Hybrid CNN + BiLSTM (Single timestep classification)

CUSTOM MODEL RESULTS
--------------------
ECR Strategy                  | Detection Module      | Precision(1) | Recall(1) | F1-score(1)
-----------------------------|------------------------|--------------|-----------|-------------
Emotion + Post Concatenation | BiLSTM                 |     0.75     |   0.58    |     0.65
+ Standardized Emotion Vectors| BiLSTM                |     0.78     |   0.63    |     0.70
+ Residual Dot Product Fusion | BiLSTM                |     0.81     |   0.57    |     0.67
Final Fusion + Hybrid Model   | CNN + BiLSTM Classifier|    0.83     |   0.55    |     0.66

TECHNOLOGIES USED
-----------------
- Python
- TensorFlow / Keras
- BERT (via HuggingFace Transformers)
- CRF Layer (for sequence tagging)
- Scikit-learn
- NLTK, regex
- Pandas, NumPy

KEY TAKEAWAYS
-------------
- Emotion labels enhance cyberbullying detection.
- Fusion strategies like residual dot product fusion improve performance.
- CNN + BiLSTM hybrid architecture showed better results than pure sequence models.
