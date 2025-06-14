# SJ-DECEN 
CYBERBULLYING DETECTION USING A CUSTOM DECEN-INSPIRED DEEP LEARNING APPROACH
==============================================================================

This repository presents a deep learning pipeline for cyberbullying detection,
inspired by the DECEN framework that was originally used for depression detection and has been adapted here to incorporate emotion-aware features for enhanced performance in cyberbullying detection. The model is trained on Wikipedia Comments which contains user-generated comments labeled for toxicity. 

DATASET
-------
Source: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data

The original dataset includes multi-label classifications (e.g., toxic, severe toxic, obscene, threat, insult, identity hate). This study simplifies the task to binary classification.

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

BACKGROUND AND RELATED WORK
---------------------------
This project is motivated by prior research in two complementary areas:
1. Antisocial Behavior Detection Using Deep Learning:
The study by Zinovyeva et al. (2020) demonstrated how deep learning architectures like CNNs, LSTMs, and GRUs could be used to detect antisocial behavior in online discussions. Their methodology was replicated to establish a performance benchmark for cyberbullying detection.

Zinovyeva, E., Härdle, W. K., & Lessmann, S. (2020). Antisocial online behavior detection using deep learning. Decision Support Systems, 138, 113362. https://doi.org/10.1016/j.dss.2020.113362

2. Emotion-Aware Deep Learning for Depression Detection:
The proposed model, SJ-DECEN, is based on the DECEN architecture introduced by Yan et al. (2025) for detecting depression from social media content. DECEN integrates token-level emotion recognition with contextual representations to improve classification accuracy. This approach was adapted to the cyberbullying domain by automating emotion labeling through the NRC Emotion Lexicon and exploring fusion strategies that better capture emotional context.

Yan, Z., Peng, F., & Zhang, D. (2025). DECEN: A deep learning model enhanced by depressive emotions for depression detection from social media content. Decision Support Systems, 114421. https://doi.org/10.1016/j.dss.2025.114421

Our contribution lies in the implementation of the DECEN based framework for cyberbullying detection, rather than modifying or extending the baseline models.

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
Standardized Emotion Vectors| BiLSTM                |     0.78     |   0.63    |     0.70
Residual Dot Product Fusion | BiLSTM                |     0.81     |   0.57    |     0.67
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
