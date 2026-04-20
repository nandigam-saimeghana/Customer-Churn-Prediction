# Customer-Churn-Prediction
Machine learning project to predict customer churn using Python and Logistic Regression
# Customer Churn Prediction

This project predicts whether a customer will churn using Machine Learning.

## 📌 Project Overview
- Data Loading and Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Logistic Regression Model
- Model Evaluation

## 📊 Results
- Accuracy: ~80-85%
- Confusion Matrix and Classification Report included

## 🛠 Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## ▶️ How to Run
1. Clone the repository
2. Run the Python file

## 📁 Dataset
Telco Customer Churn Dataset

## 📌 Author
SaiMeghana

CODE:-
1.import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

2.# Load the dataset
df = pd.read_excel(r"C:\Users\Admin\Downloads\customer_churn_prediction.xlsx")

# Show first 5 rows
df.head()

3.# Shape of dataset (rows, columns)
print("Shape:", df.shape)

# Data types and non-null counts
df.info()

# Summary statistics for numerical columns
df.describe()

4. # Count missing values per column
df.isnull().sum()

5. # Count of churn values
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Distribution")
plt.show()

6. # Example: Churn by Gender
sns.countplot(x="gender", hue="Churn", data=df)
plt.title("Churn by Gender")
plt.show()

# Example: Churn by Internet Service
sns.countplot(x="InternetService", hue="Churn", data=df)
plt.title("Churn by Internet Service")
plt.show()

7. # Monthly Charges distribution
sns.histplot(df["MonthlyCharges"], bins=30, kde=True)
plt.title("Distribution of Monthly Charges")
plt.show()

# Monthly Charges vs Churn
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

8. # Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

9. # Replace 'No internet service' and 'No phone service' with 'No'
cols_to_replace = ['MultipleLines','OnlineSecurity','DeviceProtection',
                   'TechSupport','StreamingTV','StreamingMovies']

for col in cols_to_replace:
    df[col] = df[col].replace({'No internet service':'No', 'No phone service':'No'})
    
10. # Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check again
df.info()
11. # Check missing values again
df.isnull().sum()

# Drop rows with missing TotalCharges
df = df.dropna(subset=['TotalCharges'])
12. # Convert Yes/No to 1/0
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

df_encoded.head()

13. from sklearn.model_selection import train_test_split

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

14. from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


