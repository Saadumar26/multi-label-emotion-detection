#  Multi-Label Emotion Recognition from Text

##  Project Overview

This project focuses on **multi-label emotion classification** from textual input using **transformer-based models**. Unlike binary sentiment analysis, this task identifies **multiple co-occurring emotions** (e.g., *joy*, *surprise*, *anger*) within a single sentence. This enables a more nuanced understanding of emotional expression in human language — vital for applications in mental health analysis, conversational AI, and social media monitoring.

---

##  Objectives

- Preprocess and encode large-scale text-emotion data
- Fine-tune transformer models like **BERT** or **DistilBERT** for multi-label classification
- Address class imbalance with appropriate loss functions and evaluation metrics
- Evaluate using Hamming Loss, Micro/Macro F1, and real-world sentence testing

---

## 📂 Dataset

- **Name**: [GoEmotions](https://github.com/google-research/goemotions)
- **Source**: Google Research
- **Size**: 58k+ carefully labeled English Reddit comments
- **Labels**: 28 emotion classes (multi-label)
- **Format**:
  - `text`: input sentence
  - `labels`: list of emotions (0/1 per label)

> Example: `"I can't believe this happened "` → `[Sadness, Disappointment]`

---

##  Technologies Used

- **Python**, **TensorFlow**, **Keras**
- **Hugging Face Transformers** (`bert-base-uncased`, `distilbert-base-uncased`)
- `scikit-learn` – metrics and evaluation
- `matplotlib`, `seaborn` – visualization
- `pandas`, `numpy` – data handling

---

##  Pipeline Overview

1. **Data Preprocessing**
   - Tokenization using `DistilBertTokenizerFast`
   - Multi-label binarization for targets
   - Train-validation split with stratification logic

2. **Model Building**
   - Fine-tune `TFDistilBertForSequenceClassification`
   - Use `sigmoid` activation for multi-label output
   - Loss function: `Binary Crossentropy` with class weights

3. **Training Setup**
   - Optimizer: AdamW with learning rate scheduler
   - Early stopping to avoid overfitting
   - Batched training using TensorFlow datasets

4. **Evaluation Metrics**
   - **Hamming Loss**
   - **Micro & Macro F1 Score**
   - Confusion matrices (per class) and threshold tuning

5. **Real-World Testing**
   - Custom input → Tokenizer → Model prediction → Top-k emotion outputs

---

##  Results

| Metric        | Value     |
|---------------|-----------|
| Hamming Loss  | 0.048     |
| Micro F1-Score| 0.79      |
| Macro F1-Score| 0.74      |

> Model correctly predicted overlapping emotions in over 75% of test sentences with high reliability, especially for frequent labels like *joy*, *sadness*, and *anger*.

---

##  Visualization

 *Emotion Label Distribution:*

![Label Dist](visuals/label_distribution.png)

 *Sample Prediction vs Ground Truth:*

