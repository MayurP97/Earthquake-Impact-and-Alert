# 🌍 Earthquake Impact Prediction using Machine Learning

## 📌 Project Overview

This project develops a Machine Learning model to predict earthquake impact/alert categories using historical earthquake data. The workflow covers the complete data science pipeline including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning, performance evaluation, and dashboard preparation.

The primary objective is to build an accurate classification model that can assist in predicting earthquake alert levels based on seismic attributes.

---

## 🚀 Features

- Data Cleaning & Preprocessing
- Missing Value Handling
- Duplicate Removal
- Outlier Detection & Treatment
- Feature Scaling
- Categorical Encoding
- Class Imbalance Handling
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Feature Selection
- Multiple Machine Learning Models
- Hyperparameter Tuning
- Model Performance Comparison
- Confusion Matrix Analysis
- Business Interpretation
- Streamlit Dashboard Preparation

---

## 📂 Project Structure

```
Earthquake_Impact/
│
├── Earthquake_Impact.ipynb          # Complete ML Pipeline
├── earthquake_alert_balanced_dataset.csv
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

The project uses a balanced earthquake dataset containing seismic information.

Typical features include:

- Magnitude
- Depth
- Latitude
- Longitude
- Location
- Other seismic parameters
- Alert Category (Target Variable)

Dataset used in notebook:

```
earthquake_alert_balanced_dataset.csv
```

---

# 🔄 Project Workflow

## 1. Data Cleaning

- Remove duplicate records
- Handle missing values
- Correct data types
- Detect and treat outliers

---

## 2. Data Preprocessing

- Encode categorical variables
- Standardize numerical features
- Handle class imbalance
- Remove irrelevant or leakage-prone columns

---

## 3. Exploratory Data Analysis (EDA)

Performed:

- Distribution Analysis
- Correlation Heatmaps
- Target Variable Analysis
- Outlier Visualization

Libraries used:

- Matplotlib
- Seaborn

---

## 4. Feature Engineering

Includes:

- Creating derived variables
- Feature importance analysis
- Correlation-based feature selection

---

## 5. Model Training

The following Machine Learning algorithms were implemented and compared.

### Classification Models

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- Gaussian Naive Bayes
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP)

---

## 6. Hyperparameter Tuning

Performed using:

- GridSearchCV

Models tuned:

- Random Forest
- LightGBM

---

## 7. Model Evaluation

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

---

## 8. Model Comparison

Different models are compared based on:

- Training Time
- Prediction Performance
- Accuracy
- F1 Score
- Precision
- Recall
- ROC-AUC

The best-performing model is selected after hyperparameter tuning.

---

## 9. Error Analysis

The notebook includes:

- Misclassification Analysis
- Model Limitations
- Areas for Improvement

---

## 10. Business Interpretation

The project explains how Machine Learning predictions can assist in:

- Earthquake risk assessment
- Disaster preparedness
- Early warning support
- Resource allocation
- Decision-making for emergency response

---

## 11. Dashboard

A Streamlit dashboard section is included for interactive visualization and prediction.

---

# 🛠️ Technologies Used

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost
- LightGBM

### Dashboard

- Streamlit

---

# 📦 Required Libraries

Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm streamlit
```

Or using:

```bash
pip install -r requirements.txt
```

---

# ▶️ How to Run

Clone the repository

```bash
git clone https://github.com/yourusername/Earthquake_Impact.git
```

Move to project directory

```bash
cd Earthquake_Impact
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

Open

```
Earthquake_Impact.ipynb
```

---

# 📈 Machine Learning Pipeline

```
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Data Preprocessing
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Feature Engineering
   │
   ▼
Feature Selection
   │
   ▼
Model Training
   │
   ▼
Hyperparameter Tuning
   │
   ▼
Model Evaluation
   │
   ▼
Best Model Selection
   │
   ▼
Prediction & Dashboard
```

---

# 📚 Future Improvements

- Deep Learning Models
- Real-time Earthquake API Integration
- Interactive Geographic Maps
- Automated Alert System
- Cloud Deployment
- Docker Support
- CI/CD Pipeline
- Model Monitoring

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request

---

# 📄 License

This project is intended for educational and research purposes.

---

# 👨‍💻 Author

**Mayur**

Machine Learning | Data Science | Python Developer

---

## ⭐ If you found this project useful, please consider giving it a Star on GitHub!
