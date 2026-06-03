# 🚀 RetentionIQ
# AI-Powered Customer Retention Intelligence Platform

> Predicting customer churn before it happens using Machine Learning, Explainable AI, and Business Intelligence.

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Machine%20Learning-XGBoost-success?style=for-the-badge">
  <img src="https://img.shields.io/badge/Explainable%20AI-SHAP-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge">
</p>

---

## 🌟 Project Vision

In today's competitive market, retaining an existing customer is significantly more cost-effective than acquiring a new one.

**RetentionIQ** is a data-driven customer churn prediction platform designed to help organizations identify at-risk customers before they leave. By combining advanced machine learning models, feature engineering, and explainable AI techniques, RetentionIQ transforms raw customer data into actionable business intelligence.

The platform empowers businesses to:

✅ Predict customer churn accurately

✅ Understand the reasons behind churn

✅ Identify high-risk customer segments

✅ Design proactive retention strategies

✅ Reduce customer acquisition costs

---

# 🎯 Business Problem

Customer churn is one of the most critical challenges faced by telecom, SaaS, banking, insurance, and subscription-based companies.

A company may lose millions in revenue if customers leave unexpectedly.

Traditional approaches identify churn after customers have already left.

**RetentionIQ solves this problem by predicting churn in advance**, enabling businesses to take corrective actions before customer loss occurs.

---

# 💼 Real-World Impact

Imagine a company with 1 million customers.

Even a small reduction in churn rate can save:

- Millions in annual revenue
- Marketing acquisition costs
- Customer support expenses

RetentionIQ helps businesses shift from:

```text
Reactive Decision Making
            ↓
Proactive Customer Retention
```

---

# 🚀 Key Features

### 📊 Advanced Data Analytics
- Comprehensive Exploratory Data Analysis
- Customer Behavior Analysis
- Churn Pattern Discovery

### 🧠 Machine Learning Pipeline
- Data Cleaning
- Feature Engineering
- Data Preprocessing
- Model Training
- Model Evaluation

### ⚡ Intelligent Prediction Engine
- Logistic Regression
- Random Forest
- XGBoost

### 🔍 Explainable AI
- SHAP Explainability
- Feature Importance Analysis
- Transparent Decision Making

### 📈 Business Intelligence
- Customer Retention Insights
- Strategic Recommendations
- Risk Analysis

---

# 🏗️ System Architecture

```text
Customer Data
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Data Preprocessing
      │
      ▼
Class Balancing (SMOTE)
      │
      ▼
Machine Learning Models
      │
      ▼
Model Evaluation
      │
      ▼
SHAP Explainability
      │
      ▼
Business Insights & Recommendations
```

---

# 📂 Dataset Overview

### Dataset Used

IBM Telco Customer Churn Dataset

### Dataset Statistics

| Metric | Value |
|----------|----------|
| Total Records | 7,043 |
| Features | 21 |
| Domain | Telecom |
| Target Variable | Churn |

### Prediction Target

| Value | Meaning |
|---------|---------|
| Yes | Customer Churned |
| No | Customer Retained |

---

# 🔬 Methodology

## Phase 1 — Data Understanding

Understanding customer demographics, subscriptions, payment methods, tenure, and service usage patterns.

---

## Phase 2 — Data Cleaning

Performed:

- Missing Value Handling
- Data Type Conversion
- Duplicate Validation
- Feature Selection

---

## Phase 3 — Exploratory Data Analysis

Analyzed:

- Churn Distribution
- Contract Types
- Monthly Charges
- Customer Tenure
- Payment Methods
- Service Usage Patterns

Key objective:

> Discover hidden churn-driving patterns within customer behavior.

---

## Phase 4 — Feature Engineering

Created meaningful features such as:

### Average Spending

```python
AvgCharges = TotalCharges / Tenure
```

### Long-Term Customer Indicator

```python
LongTermCustomer
```

### High Value Customer Indicator

```python
HighValueCustomer
```

These engineered features improved model learning capability.

---

## Phase 5 — Data Preprocessing

Implemented:

- Label Encoding
- Train-Test Split
- Feature Transformation

---

## Phase 6 — Class Imbalance Handling

Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance churn classes and improve predictive performance.

---

## Phase 7 — Model Development

The following machine learning models were trained and evaluated:

| Model | Purpose |
|---------|---------|
| Logistic Regression | Baseline Model |
| Random Forest | Ensemble Learning |
| XGBoost | Final Optimized Model |

---

## Phase 8 — Explainable AI

Implemented SHAP to answer:

### Why is a customer likely to churn?

### Which features contribute most to churn?

### How can retention teams intervene?

This transforms the model from a black box into a transparent decision-support system.

---

# 📊 Evaluation Metrics

Performance measured using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

These metrics ensure reliable and business-relevant model evaluation.

---

# 💡 Key Business Insights

### Insight #1

Customers with Month-to-Month contracts are significantly more likely to churn.

### Insight #2

Customers with shorter tenure exhibit higher churn probability.

### Insight #3

Higher monthly charges increase customer attrition risk.

### Insight #4

Long-term customers demonstrate stronger loyalty and retention.

### Insight #5

Customer retention strategies should focus on early-stage subscribers.

---

# 📈 Strategic Recommendations

### Customer Retention Campaigns

Target customers predicted as high-risk.

### Loyalty Programs

Reward long-term subscribers.

### Contract Conversion

Encourage customers to switch from month-to-month to annual plans.

### Personalized Offers

Provide customized discounts based on churn probability.

### Customer Engagement

Improve onboarding experiences for new subscribers.

---

# 🛠️ Technology Stack

## Programming

- Python

## Data Analysis

- Pandas
- NumPy

## Visualization

- Matplotlib
- Seaborn

## Machine Learning

- Scikit-Learn
- XGBoost
- Random Forest
- Logistic Regression

## Explainable AI

- SHAP

## Model Serialization

- Joblib

---

# 📁 Project Structure

```text
RetentionIQ/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_explainability.ipynb
│
├── models/
│   └── best_model.pkl
│
├── reports/
│   └── visuals/
│
├── src/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Future Scope

- Real-Time Churn Prediction
- Customer Segmentation Engine
- Deep Learning Models
- Interactive Dashboard
- Automated Retention Recommendation System
- Cloud Deployment

---

# 👨‍💻 Project Team

### Harsh Sharma
**Machine Learning Engineer & Data Analyst**

- Data Analysis
- Feature Engineering
- Model Development
- Business Insights

---

### Priya Rani
**Machine Learning Engineer & Research Analyst**

- Data Preprocessing
- Model Evaluation
- Explainable AI
- Documentation & Research

---

# 🏆 Skills Demonstrated

✅ Machine Learning

✅ Data Analytics

✅ Feature Engineering

✅ Exploratory Data Analysis

✅ Explainable AI

✅ Business Intelligence

✅ Predictive Modeling

✅ Team Collaboration

✅ Problem Solving

---

# 🌍 Conclusion

RetentionIQ demonstrates how Artificial Intelligence can be leveraged to transform customer retention strategies through predictive analytics and explainable machine learning.

By identifying churn risks early and uncovering the drivers behind customer attrition, businesses can make smarter decisions, improve customer satisfaction, and maximize long-term revenue growth.

---

## ⭐ If you found this project interesting, don't forget to star this repository and connect with us!

### Built with ❤️ by Harsh Sharma & Priya Rani
