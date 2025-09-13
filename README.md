# Pneumonia Detection from Chest X-Ray Images Using Deep Learning

## Overview
This project develops a deep learning model to classify chest X-ray images as **NORMAL** or **PNEUMONIA**, addressing automated disease diagnosis in healthcare. Built with **TensorFlow/Keras** and **MobileNetV2**, it includes data preprocessing, hyperparameter tuning, class imbalance handling, model explainability (**Grad-CAM**), and a Flask API simulation.  


---

## Key Features
- **Dataset:** Subsampled Kaggle Chest X-Ray Pneumonia dataset (500 train, 100 val, 624 test images).
- **Model:** MobileNetV2 with tuned hyperparameters (units, dropout, learning rate) and class weights.
- **Performance:** ~86–90% test accuracy, ~86% precision, ~92% recall for PNEUMONIA.
- **Explainability:** Grad-CAM generates heatmaps to visualize regions (e.g., lung consolidations) driving model predictions, ensuring clinical relevance and trust.
- **Deployment:** Flask API for real-time inference, cloud-ready.
- **Tools:** Python, TensorFlow, Keras Tuner, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, OpenCV.

---

## Project Motivation
This project showcases expertise in healthcare-focused ML, addressing pneumonia detection with advanced techniques like hyperparameter tuning, class weighting, and explainability.

---

## Technical Details
- **Dataset:**  
  - 250 train + 50 val images per class (NORMAL, PNEUMONIA) to fit Colab (~12GB RAM).  
  - Test set: 624 images (234 NORMAL, 390 PNEUMONIA).  

- **Preprocessing:**  
  - Images resized to 128×128.  
  - Augmentation (rotation, zoom, flip).  

- **Model:**  
  - MobileNetV2 (pre-trained) with custom head.  
  - Tuned via Keras Tuner (units: 32–128, dropout: 0.3–0.7, learning rate: 1e-4 to 1e-2).  

- **Class Imbalance:**  
  - Weights: `1.5` for NORMAL, `1.0` for PNEUMONIA to improve NORMAL recall.  

- **Training:**  
  - 3 epochs post-tuning (~8–10 mins on GPU).  

- **Explainability:**  
  - Grad-CAM uses the `block_16_project` layer (or `Conv_1` fallback) in MobileNetV2 to compute gradient-based heatmaps, highlighting lung regions critical to predictions. This ensures interpretability for clinicians, reduces diagnostic errors, and aligns with regulatory standards (e.g., FDA for AI diagnostics). 

- **Evaluation:**  
  - Accuracy, precision, recall, F1-score, confusion matrix.  

- **Runtime:**  
  - ~15–20 mins for training/evaluation.  
  - ~2–5 mins for Grad-CAM/API (GPU).  

---

## Results
- **Training:** Validation accuracy ~90–92% after 3 epochs.  

- **Test Metrics:**  
  - Accuracy: ~86–90%  
  - Precision: ~86% (PNEUMONIA), ~85% (NORMAL)  
  - Recall: ~92% (PNEUMONIA), ~80–85% (NORMAL with weights)  
  - F1-Score: ~89% (PNEUMONIA), ~82% (NORMAL)  

- **Outputs:**  
  - Class distribution plots  
  - Training curves  
  - Confusion matrix  
  - Grad-CAM heatmaps  


### Prerequisites
- Google Colab with GPU (`Runtime > Change runtime type > GPU`)  
- Kaggle API key (`kaggle.json`)  
- Python 3.12, TensorFlow, Keras Tuner, OpenCV, Flask  

