# Machine Learning Portfolio: Classification, Regression & Clustering

This repository contains end-to-end machine learning workflows demonstrating data preprocessing, exploratory data analysis (EDA), and model implementation across supervised and unsupervised learning tasks.

## 📂 Project Summaries

### 1. Heart Disease Prediction (`.\Classification\prj_1.ipynb`, `.\Classification\script.ipynb`)
**Objective:** Predict the presence of heart disease using clinical patient data.
* **Algorithms Used:** Logistic Regression & K-Nearest Neighbors (KNN).
* **Key Findings:** * **Logistic Regression** achieved a **Recall of 88.89%**, making it superior for medical screening by minimizing missed cases (False Negatives).
    * **Optimized KNN** (tuned via GridSearchCV) showed higher **Precision (66.67%)** but struggled with a low **Recall (44.44%)**.
* **Tech Stack:** `scikit-learn`, `GridSearchCV`, `StandardScaler`.

### 2. Medical Insurance Cost Regression (`.\insurance\script.ipynb`)
**Objective:** Estimate annual medical charges based on demographic and lifestyle factors.
* **Algorithm:** Linear Regression.
* **Key Results:** * **R² Score:** 0.81 (The model explains 81% of cost variance).
    * **MAE:** 4181.35 (Average prediction error).
* **Insights:** Large prediction errors (RMSE = 5966.98) suggest outliers in high-cost insurance cases.

### 3. Iris Species Clustering (`.\Clustering\script.ipynb` - Iris version)
**Objective:** Group iris flowers into clusters based on physical measurements without using labels.
* **Algorithm:** K-Means Clustering ($k=3$).
* **Key Results:** * **Adjusted Rand Index (ARI):** 0.62 (Confirmed strong similarity to true species labels).
    * **Silhouette Score:** 0.46 (Indicates decent separation between clusters).
* **Techniques:** Feature scaling with `StandardScaler` was essential for cluster accuracy.

---

## 📊 Performance Comparison (Heart Disease)

In medical diagnostics, missing a sick patient (False Negative) is more critical than a false alarm.

| Metric | Logistic Regression | Optimized KNN |
| :--- | :--- | :--- |
| **Accuracy** | 81.36% | 76.27% |
| **Recall (Sensitivity)** | **88.89%** | 44.44% |
| **Precision** | 64.00% | **66.67%** |
| **F1-Score** | 74.42% | 53.33% |



---

## 🛠️ Data Preprocessing Pipeline

* **Handling Missing Values:** Replaced non-standard characters (like `'?'`) with `0` or medians and performed type conversion.
* **Feature Engineering:** Performed categorical mapping for features like `sex`, `smoker`, and `region`.
* **Automated EDA:** Utilized `ydata-profiling` to generate comprehensive HTML reports for data inspection.

---

## 🚀 How to Run

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/Rawane-dev/Evaluation-Metrics.git](https://github.com/Rawane-dev/Evaluation-Metrics.git)
2. **Install Dependencies:**
   ```bash
   pip install pandas numpy scikit-learn ydata-profiling
3. **Execute Notebooks:**
    Open any .ipynb file to view the step-by-step analysis. Ensure data.csv and insurance.csv are in the root directory.
