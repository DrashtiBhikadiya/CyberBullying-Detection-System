# CyberBullying-Detection-System
Automated toxicity detection using RoBERTa &amp; DistilBERT. Trained on 47,000+ entries with ~87% accuracy





# 🛡️ Cyberbullying Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-HuggingFace%20Transformers-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📄 Overview
Social media platforms are witnessing a massive surge in toxic content. This project is a **Cyberbullying Detection System** designed to automatically classify text into varying categories of toxicity. 

Leveraging state-of-the-art Deep Learning models (**RoBERTa** and **DistilBERT**), the system analyzes text to identify harmful intent with high precision. The model treats this as a multi-class classification problem rather than a simple binary (bully/not bully) task.

## 📊 Dataset
The model was trained on a comprehensive dataset consisting of **47,000+ entries**.
* **Source:** [Mention source if known, e.g., Kaggle/Twitter API]
* **Classes:** The data is categorized into 6 distinct classes:
    * *Age*
    * *Ethnicity*
    * *Gender*
    * *Religion*
    * *Other Cyberbullying*
    * *Not Cyberbullying*
    

## 🛠️ Methodology & Pipeline
The project follows a rigorous NLP pipeline:

1.  **Data Preprocessing:** Cleaning text (removing URLs, special characters, lowercasing).
2.  **Tokenization:** Using specific tokenizers suited for Transformer models.
3.  **Feature Engineering:** Implementation of TF-IDF for baseline analysis. 
4.  **Model Training:** Fine-tuning pre-trained Transformer models on the dataset.
5.  **Evaluation:** Comparing accuracy and loss metrics.

## 📈 Model Performance
We experimented with two powerful Transformer-based models. Both models achieved competitive accuracy, demonstrating the robustness of the approach.

-----------------------------------------
            ROBERTA MODEL
-----------------------------------------

Classification report:
                     precision    recall  f1-score   support

                age       0.99      0.98      0.99      1598
          ethnicity       0.98      0.97      0.98      1592
             gender       0.89      0.92      0.90      1595
  not_cyberbullying       0.76      0.60      0.67      1589
other_cyberbullying       0.69      0.82      0.75      1565
           religion       0.96      0.97      0.97      1600

           accuracy                           0.88      9539
          macro avg       0.88      0.88      0.88      9539
       weighted avg       0.88      0.88      0.88      9539

-------------------------------------------------
                DISTILBERT MODEL
-------------------------------------------------      
      Classification report:
                     precision    recall  f1-score   support

                age       0.99      0.98      0.99      1598
          ethnicity       0.98      0.98      0.98      1592
             gender       0.89      0.91      0.90      1595
  not_cyberbullying       0.72      0.61      0.66      1589
other_cyberbullying       0.68      0.78      0.73      1565
           religion       0.96      0.97      0.97      1600

           accuracy                           0.87      9539
          macro avg       0.87      0.87      0.87      9539
       weighted avg       0.87      0.87      0.87      9539
      

## 💻 Tech Stack
* **Language:** Python
* **Libraries:**
    * `Transformers` (Hugging Face)
    * `Scikit-learn` (for TF-IDF and metrics)
    * `Pandas` & `NumPy` (Data manipulation)
    * `PyTorch` / `TensorFlow` (Deep Learning backend)
    * `Matplotlib` / `Seaborn` (Visualization)

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/DrashtiBhikadiya/CyberBullying-Detection-System]
    cd cyberbullying-detection
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

To test the model on custom text or run the training pipeline:

```bash
# Example: To run the prediction script
streamlit run app.py 

