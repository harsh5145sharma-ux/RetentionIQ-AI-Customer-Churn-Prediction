# 🚀 RetentionIQ: AI-Powered Customer Churn Prediction & Business Intelligence System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-green)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Business Intelligence](https://img.shields.io/badge/Business%20Intelligence-Analytics-orange)

> An end-to-end Machine Learning and Business Intelligence solution designed to predict customer churn, identify high-risk customers, and generate actionable retention insights through advanced analytics, predictive modeling, and customer segmentation.

---

# 📌 Project Overview

Customer retention is one of the most important factors driving business growth and profitability. Losing existing customers can significantly impact revenue, customer lifetime value, and acquisition costs.

**RetentionIQ** leverages Machine Learning, Predictive Analytics, and Business Intelligence techniques to proactively identify customers at risk of churn and provide organizations with data-driven retention strategies.

The project combines data preprocessing, exploratory data analysis, predictive modeling, customer risk segmentation, model explainability, and business intelligence reporting into a complete end-to-end workflow.

---

# 🎯 Business Objective

The primary goals of this project are:

- Predict customers likely to churn before they leave.
- Identify the most influential churn-driving factors.
- Segment customers based on churn risk levels.
- Support targeted customer retention campaigns.
- Enable data-driven business decision making.
- Improve customer lifetime value through proactive intervention.

---

# 📊 Dataset Information

### Dataset

**RetentionIQ Customer Churn Dataset**

### Target Variable

```text
churned
```

- 1 = Customer Churned
- 0 = Customer Retained

### Features Included

- Age
- Gender
- Region
- Subscription Plan
- Contract Type
- Payment Method
- Tenure Months
- Monthly Charges
- Total Charges
- Login Activity
- Features Used
- Support Tickets
- NPS Score
- Payment Failures
- Email Open Rate
- Last Login Activity

---

# 🔄 End-to-End Workflow

## 1️⃣ Exploratory Data Analysis (EDA)

Performed comprehensive exploratory analysis to understand customer behavior, engagement patterns, and churn trends.

### EDA Dashboard Includes

✅ Churn Distribution Analysis

✅ Age Distribution by Churn Status

✅ Tenure Analysis

✅ Monthly Charges Analysis

✅ Contract Type Churn Trends

✅ Payment Failure Impact Analysis

✅ NPS Score Analysis

✅ Plan-wise Churn Analysis

✅ Login Activity Analysis

### Output Generated

```text
01_eda_dashboard.png
```

---

## 2️⃣ Data Preprocessing

Built a complete preprocessing pipeline to prepare the data for machine learning.

### Steps Performed

- Feature Selection
- Missing Value Handling
- Median Imputation
- Label Encoding
- Train-Test Splitting
- Feature Scaling
- Data Validation

### Techniques Used

- LabelEncoder
- SimpleImputer
- StandardScaler

---

## 3️⃣ Machine Learning Model Development

Developed and compared multiple classification algorithms to identify the most effective churn prediction model.

### Models Implemented

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

---

## 4️⃣ Model Evaluation & Comparison

Evaluated all models using industry-standard performance metrics.

### Metrics Used

- Accuracy Score
- ROC-AUC Score
- Cross Validation Score
- Classification Report
- Confusion Matrix
- ROC Curve Analysis

### Output Generated

```text
02_model_comparison.png
```

Includes:

- Accuracy Comparison
- ROC-AUC Comparison
- Cross Validation Analysis
- ROC Curves for All Models

---

## 5️⃣ Best Model Deep Dive

Performed advanced analysis on the highest-performing model.

### Analysis Includes

- Confusion Matrix
- Feature Importance Analysis
- Probability Distribution Analysis
- Churn Prediction Behavior

### Output Generated

```text
03_best_model_deepdive.png
```

---

## 6️⃣ Customer Risk Segmentation

Converted model predictions into actionable business insights by categorizing customers into risk groups.

### Risk Categories

| Segment | Description |
|----------|-------------|
| Low Risk | Highly Retainable Customers |
| Medium Risk | Moderate Churn Probability |
| High Risk | Immediate Retention Priority |

---

## 7️⃣ Business Intelligence & Lift Analysis

Implemented cumulative gains and lift analysis to support retention campaign planning.

### Insights Generated

- High-Risk Customer Identification
- Customer Prioritization Strategy
- Retention Campaign Optimization
- Churn Capture Efficiency Analysis

### Output Generated

```text
04_business_insights.png
```

---

# 📈 Key Business Insights

The analysis revealed that customer churn is strongly influenced by:

- Low Customer Engagement
- Reduced Login Activity
- Low NPS Scores
- High Payment Failures
- Short Customer Tenure
- Subscription Plan Type
- Contract Type Selection

These insights can help organizations proactively retain customers before churn occurs.

---

# 🛠️ Tech Stack

### Programming & Analytics

- Python
- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-Learn

### ML Techniques

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

### Evaluation Techniques

- ROC-AUC Analysis
- Cross Validation
- Confusion Matrix
- Classification Report
- Lift Analysis
- Risk Segmentation

---

# 📁 Project Structure

```text
RetentionIQ-AI-Customer-Churn-Prediction
│
├── Main/
│   └── RetentionIQ_churn_prediction.py
│
├── data/
│   ├── retention_iq_churn_dataset.csv
│   └── retention_iq_churn_dataset_cleaned.csv
│
├── images/
│   ├── 01_eda_dashboard.png
│   ├── 02_model_comparison.png
│   ├── 03_best_model_deepdive.png
│   └── 04_business_insights.png
│
├── models/
│   ├── best_model.pkl
│   ├── label_encoders.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── .gitignore
├── LICENSE
├── README.md
├── RetentionIQ_churn_prediction.py
└── requirements.txt
```

---

# 🤖 Model Artifacts

The project stores reusable machine learning components inside the `models/` directory.

| File | Description |
|---------|-------------|
| best_model.pkl | Best-performing churn prediction model selected after evaluation |
| label_encoders.pkl | Encoders used for categorical feature transformation |
| scaler.pkl | StandardScaler used during feature preprocessing |

These artifacts enable future predictions without retraining the entire machine learning pipeline.

---

# 📊 Generated Outputs

| Output File | Description |
|------------|-------------|
| 01_eda_dashboard.png | Multi-panel exploratory analysis dashboard |
| 02_model_comparison.png | Model comparison and ROC analysis |
| 03_best_model_deepdive.png | Confusion matrix and feature importance analysis |
| 04_business_insights.png | Risk segmentation and lift analysis dashboard |

---

# 🌟 Project Highlights

✔ End-to-End Machine Learning Pipeline

✔ Automated Exploratory Data Analysis

✔ Multiple Model Comparison Framework

✔ Customer Risk Segmentation

✔ Business Intelligence Reporting

✔ Lift & Gains Analysis

✔ Feature Importance Analysis

✔ Model Persistence using PKL Artifacts

✔ Real-World Customer Retention Use Case

✔ Production-Ready Repository Structure

---

# 🚀 Future Enhancements

- XGBoost & LightGBM Integration
- Hyperparameter Tuning
- SHAP Explainability
- Streamlit Dashboard Deployment
- Real-Time Prediction API
- Automated Retention Recommendation Engine

---

# 👥 Contributors

## Harsh Kumar Sharma

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Machine Learning Model Development
- Model Evaluation & Optimization

## Priya Rani

- Business Insights Generation
- Data Storytelling & Interpretation
- Visualization Assets Management
- Presentation Design
- Project Documentation

---

# 📜 License

This project is licensed under the MIT License.

---

## ⭐ If you found this project useful, consider giving it a star.

**Built with Machine Learning, Data Analytics, and Business Intelligence to transform customer data into actionable retention strategies.**
