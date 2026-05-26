# Employee Attrition Prediction using Machine Learning & Deep Learning

An end-to-end Employee Attrition Prediction project focused on analyzing employee behavior, identifying attrition factors, and building predictive models using Machine Learning and Deep Learning techniques.

---

# Project Overview

Employee attrition is a major challenge for organizations. This project aims to:

- Analyze employee-related factors affecting attrition
- Perform Exploratory Data Analysis (EDA)
- Handle categorical and numerical preprocessing
- Handle imbalanced data using SMOTE
- Train multiple Machine Learning models
- Build a Deep Learning model using TensorFlow/Keras
- Compare models using ROC-AUC and classification metrics

---

# Dataset Information

The dataset contains HR-related employee information such as:

- Department
- Education
- Job Satisfaction
- Environment Satisfaction
- Work-Life Balance
- Performance Rating
- Overtime
- Years at Company
- Monthly Income
- Attrition Status

### Target Variable

`Attrition`

- Yes → Employee Left
- No → Employee Stayed

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost
- TensorFlow / Keras

---

# Data Cleaning & Feature Engineering

## Removed Unnecessary Columns

```python
df.drop('EmployeeCount', axis=1, inplace=True)
df.drop('StockOptionLevel', axis=1, inplace=True)
df.drop('StandardHours', axis=1, inplace=True)
df.drop('DailyRate', axis=1, inplace=True)
df.drop('EmployeeNumber', axis=1, inplace=True)
df.drop('HourlyRate', axis=1, inplace=True)
df.drop('Over18', axis=1, inplace=True)
df.drop('TrainingTimesLastYear', axis=1, inplace=True)
```

## Encoded Binary Features

```python
df['Attrition'] = df['Attrition'].map({'Yes':1,'No':0})
df['OverTime'] = df['OverTime'].map({'Yes':1,'No':0})
df['Gender'] = df['Gender'].map({'Male':1,'Female':0})
```

## Missing Values Check

```python
df.isnull().sum().sum()
```

Output:

```python
0
```

---

# Exploratory Data Analysis (EDA)

## Attrition Distribution

```python
sns.countplot(data=df, x='Attrition')
plt.show()
```

### Insights

- Employees Left: **16.12%**
- Employees Staying: **83.88%**

---

## Department Analysis

```python
sns.countplot(data=df, x='Department')
plt.show()
```

---

## Education Analysis

```python
sns.countplot(data=df, x='Education')
plt.title('1:Below College || 2:College || 3:Bachelor || 4:Masters || 5:Doctorate')
plt.show()
```

---

## Years at Company Distribution

```python
sns.histplot(data=df, x='YearsAtCompany', bins=7)
plt.show()
```

---

## Satisfaction Metrics

Analyzed:

- Job Satisfaction
- Job Involvement
- Environment Satisfaction
- Relationship Satisfaction
- Work-Life Balance

These features showed strong relationships with attrition.

---

# Data Preprocessing Pipeline

Used:

- `ColumnTransformer`
- `Pipeline`
- `OneHotEncoder`
- `StandardScaler`
- `SimpleImputer`

```python
cat_transformer = Pipeline([
    ('cat_impute', SimpleImputer(strategy='most_frequent')),
    ('cat_encoder', OneHotEncoder(handle_unknown='ignore'))
])

num_transformer = Pipeline([
    ('num_impute', SimpleImputer(strategy='mean')),
    ('num_encoder', StandardScaler())
])
```

---

# Handling Imbalanced Data

Used SMOTE for balancing attrition classes.

```python
SMOTE(random_state=13)
```

---

# Machine Learning Models

## 1. Logistic Regression

### Performance

- Best CV Score: **0.788**
- ROC-AUC Score: **0.731**

---

## 2. XGBoost Classifier

### Performance

- Best CV Score: **0.865**
- ROC-AUC Score: **0.668**

---

## 3. Gradient Boosting Classifier

### Performance

- Best CV Score: **0.868**
- ROC-AUC Score: **0.651**

---

# Deep Learning Model

Built an Artificial Neural Network using TensorFlow/Keras.

## Model Architecture

```python
model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),

    layers.Dense(1, activation='sigmoid')
])
```

## Model Compilation

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        keras.metrics.AUC(name='auc'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')
    ]
)
```

---

# Class Imbalance Handling in Deep Learning

Applied class weights using:

```python
compute_class_weight(class_weight='balanced')
```

---

# Deep Learning Results

## Classification Report

```python
              precision    recall  f1-score   support

           0       0.88      0.97      0.93       371
           1       0.70      0.33      0.45        70

    accuracy                           0.87       441
```

## ROC-AUC Score

```python
0.8328
```

The Deep Learning model achieved the best overall ROC-AUC performance.

---

# Model Evaluation

Generated:

- ROC Curve
- Precision-Recall Curve

```python
RocCurveDisplay.from_predictions(y_test, y_pred_prob)
PrecisionRecallDisplay.from_predictions(y_test, y_pred_prob)
```

---

# Key Insights

- Employee attrition was relatively low but significant.
- Work-life balance and satisfaction metrics strongly influenced attrition.
- SMOTE improved minority class learning.
- Deep Learning outperformed traditional ML models in ROC-AUC score.
- Recall for attrition prediction remains challenging due to class imbalance.

---

# Future Improvements

- Deploy using Flask or Streamlit
- Add Explainable AI (SHAP/LIME)
- Use advanced ensemble methods
- Improve recall with threshold tuning
- Experiment with Transformer-based tabular models

---

# Project Structure

```bash
├── data/
├── notebooks/
├── models/
├── employee_attrition_prediction.ipynb
├── requirements.txt
└── README.md
```

---

# Installation

```bash
git clone <repository-url>

cd employee-attrition-prediction

pip install -r requirements.txt
```

---

# Run Project

```bash
jupyter notebook
```

Open:

```bash
employee_attrition_prediction.ipynb
```

---

# Skills Demonstrated

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Data Visualization
- Imbalanced Data Handling
- Machine Learning
- Hyperparameter Tuning
- Deep Learning
- Model Evaluation
- Pipeline Building
- Cross Validation

---

# Author

Harshit Saini

Aspiring AI/ML Engineer & Data Science Enthusiast passionate about building intelligent systems using Machine Learning, Deep Learning, and Neural Networks.
