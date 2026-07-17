# AI-Based Student Dropout Prediction Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aadi189/student-dropout-prediction/blob/main/ai_based_dropout_prediction.ipynb)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![Status](https://img.shields.io/badge/status-Grand%20Finalist%20Grade-success)

An end-to-end Machine Learning pipeline engineered to predict student dropout risk early in the academic lifecycle. By identifying vulnerable students using socio-economic, demographic, and macro-economic factors alongside multi-semester academic performance data, this system enables educational institutions to execute timely, targeted interventions.

---

## 🎯 The Problem & Impact

Educational institutional attrition is a critical global challenge affecting institutional ranking, resource allocation, and, most importantly, student livelihoods. 

Traditional intervention methods rely on end-of-year failures—**when it is already too late**. This project shifts the paradigm from **reactive tracking to proactive prediction**, analyzing patterns across 34 core features to catch high-risk students before their final evaluations.

---

## 📊 Dataset & Insights

The system processes an extensive dataset of **4,424 student records** across **35 feature dimensions** with zero missing values. Key operational insights discovered during Exploratory Data Analysis (EDA):

* **The Gender Attrition Gap:** While the baseline cohort is 64.8% female, male students exhibit a significantly higher relative dropout rate (~45.1%) compared to female peers (~25.1%).
* **Financial Stressors:** Unpaid tuition fees and outstanding student debts emerged as strong early indicators of academic friction and imminent dropout.
* **Academic Thresholds:** Curricular units enrolled vs. approved in the 1st and 2nd semesters represent the strongest mathematical predictors of success.

---

## 🛠️ Technical Architecture & Workflow

### 1. Preprocessing & Feature Engineering
* **Binary Target Mapping:** Enrolled/Graduate categories are structurally engineered against the absolute `Dropout` risk state.
* **Feature Normalization:** Feature scales are unified via standard scaling to eliminate algorithmic bias toward higher magnitude features.

### 2. Algorithmic Suite (Under Evaluation)
The pipeline imports and prepares a multi-model architecture for strict cross-evaluation:
* **Linear Baselines:** Logistic Regression (with Hyperparameter Tuning via GridSearchCV)
* **Tree-Based Ensemble Methods:** Decision Trees & Random Forest Classifiers
* **Distance-Based & Kernel Methods:** Support Vector Classifiers (SVC) & K-Nearest Neighbors (KNN)
* **Dimensionality Reduction:** Principal Component Analysis (PCA) configured for high-dimensional spatial compression.

---

## 🚀 How to Run the Project

### Option A: Instant Interactive Execution (Recommended)
Click the **Open in Colab** badge at the very top of this page to launch the notebook instantly in a cloud environment with pre-configured GPU/CPU runtimes.

### Option B: Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Aadi189/student-dropout-prediction.git
   cd student-dropout-prediction

2. Install the production stack:
   ```bash
   pip install numpy pandas matplotlib seaborn plotly scikit-learn tqdm

3. Open and execute ai_based_dropout_prediction.ipynb via Jupyter Notebook or VS Code.

   Once you replace that section, scroll down to the bottom of the page and click the green       **Commit changes** button. Everything will look perfectly formatted on your main repository    page!


📈 Planned Roadmap & Next Steps
[x] Complete Exploratory Data Analysis (EDA) and Data Wrangling.

[ ] Execute Model Training pipeline across all 5 imported classifiers.

[ ] Implement ROC-AUC, Precision-Recall curves, and Confusion Matrix evaluations to handle class distributions.

[ ] Develop a lightweight deployment wrapper (Streamlit/Gradio) for real-world institutional testing.
