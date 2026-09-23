# Global Terrorism Impact Predictor (GTIP)

This repository contains a dual-pipeline machine learning project that analyzes the Global Terrorism Database (2000–2017). It serves as a comprehensive capstone for foundational AI/ML concepts, demonstrating end-to-end data processing, model training, and performance visualization for both classification and regression tasks.

## 🧠 Core ML Concepts Applied

This project synthesizes the complete foundational machine learning workflow:

* **Maths for Machine Learning:** Applying linear algebra and statistical formulas underlying distance metrics (Euclidean) and error calculations.
* **AI vs ML vs DL:** Focusing purely on classical Supervised Machine Learning algorithms without relying on deep neural networks.
* **Core ML Topics:** Defining feature matrices (`X`) and target vectors (`y`), establishing model architectures, and generating predictions (`y_pred`).
* **Data Preprocessing:** 
  * Handling missing data (`dropna`).
  * Feature encoding (converting categorical text like `region_txt` into numerical data via One-Hot Encoding).
  * Outlier filtration to stabilize regression targets.
  * Train/Test Splitting to prevent data leakage.
* **Supervised Machine Learning:**
  * **Classification:** Using a Decision Tree Classifier (with `max_depth=5`) to predict categorical outcomes without requiring feature scaling (unlike distance-based KNN).
  * **Regression:** Using a Linear Regression model to predict continuous numerical values.
* **Model Evaluation & Metrics:** 
  * Classification: Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
  * Regression: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-Squared ($R^2$).

## 📊 Project Architecture

### 1. Classification Pipeline: Predicting Attack "Success"
* **Target:** `success` (Binary: 0 = Fail, 1 = Success).
* **Model:** Decision Tree Classifier.
* **Features Used:** `suicide`, `attacktype1_txt`, `region_txt`.
* **Performance Insights:** The model heavily biases toward predicting "Success" due to class imbalance in the historical data. It achieves high recall (catching almost all successful attacks) but generates a significant number of False Positives (predicting a failed attack would succeed).

### 2. Regression Pipeline: Predicting Fatalities
* **Target:** `nkill` (Continuous: Number of deaths, filtered to < 50 for baseline stability).
* **Model:** Linear Regression.
* **Features Used:** `success`, `suicide`, `attacktype1_txt`, `region_txt`.
* **Performance Insights:** The scatter plot analysis reveals underfitting. The model effectively estimates low-casualty events (0-10 deaths) but fails to capture the variance of mass-casualty outliers, flatlining its predictions regardless of the actual severity.

## 🚀 How to Run Locally

### Prerequisites
Ensure you have Python installed along with the required data science libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
