"""
Exploratory Data Analysis (EDA) for Telco Customer Churn Dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Update the path if your CSV is in a different location
DATA_PATH = 'data/Telco-Customer-Churn.csv'
df = pd.read_csv(DATA_PATH)

# 1. Dataset shape and info
print(f"Dataset shape: {df.shape}")
df.info()

# 2. Check missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 3. Churn distribution (countplot)
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, palette='Set2')
plt.title('Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.show()

# Business Insight:
# The plot above shows the proportion of customers who have churned vs. those who have stayed. This helps assess class imbalance, which is important for model selection and evaluation.

# 4. Churn vs Contract Type
plt.figure(figsize=(8, 5))
sns.countplot(x='Contract', hue='Churn', data=df, palette='Set1')
plt.title('Churn by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Count')
plt.legend(title='Churn')
plt.show()

# Business Insight:
# Customers with month-to-month contracts have a much higher churn rate compared to those with longer-term contracts. This suggests that contract type is a strong predictor of churn.

# 5. Churn vs Tenure
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='tenure', hue='Churn',
             multiple='stack', bins=30, palette='Set2')
plt.title('Churn by Tenure')
plt.xlabel('Tenure (months)')
plt.ylabel('Number of Customers')
plt.show()

# Business Insight:
# Customers with shorter tenure are more likely to churn. Retention strategies could focus on new customers to reduce early churn.

# 6. Churn vs Monthly Charges
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True,
            common_norm=False, palette='Set1', alpha=0.5)
plt.title('Churn by Monthly Charges')
plt.xlabel('Monthly Charges')
plt.ylabel('Density')
plt.show()

# Business Insight:
# Customers with higher monthly charges show a higher tendency to churn. Pricing strategies or value-added services could help retain these customers.

# 7. Correlation heatmap for numerical columns
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(8, 6))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap (Numerical Features)')
plt.show()

# Business Insight:
# The heatmap reveals relationships between numerical features. For example, tenure and total charges are highly correlated, which is expected. Understanding these relationships helps in feature selection and engineering.
