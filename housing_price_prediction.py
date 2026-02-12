#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Housing Price Prediction with Linear Regression
Machine Learning Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ======================================================================
# # Housing Price Prediction with Linear Regression
# ======================================================================

# ======================================================================
# ## Objective
# Build, evaluate, and interpret a linear regression model to predict housing prices.
# ======================================================================

# ======================================================================
# ## 1. Data Exploration
# ======================================================================

df = pd.read_csv('housing_data.csv')


print("First few rows of the dataset:")
print(df.head())


print("\nDataset shape:", df.shape)
print("\nDataset info:")
print(df.info())


print("\nStatistical summary:")
print(df.describe())


print("\nMissing values:")
print(df.isnull().sum())


plt.figure(figsize=(12, 8))
for i, col in enumerate(df.columns, 1):
    plt.subplot(2, 3, i)
    plt.hist(df[col], bins=20, edgecolor='black')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# ======================================================================
# ## 2. Model Training
# ======================================================================

X = df.drop('price', axis=1)
y = df['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")


model = LinearRegression()
model.fit(X_train, y_train)


print("\n--- Model Coefficients ---")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"\nIntercept: {model.intercept_:.2f}")


# ======================================================================
# ## 3. Model Evaluation
# ======================================================================

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

print("\n--- Training Set Performance ---")
print(f"MSE: {train_mse:.2f}")
print(f"RMSE: {train_rmse:.2f}")
print(f"R² Score: {train_r2:.4f}")

print("\n--- Test Set Performance ---")
print(f"MSE: {test_mse:.2f}")
print(f"RMSE: {test_rmse:.2f}")
print(f"R² Score: {test_r2:.4f}")


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Training Set: Predicted vs Actual')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Test Set: Predicted vs Actual')
plt.grid(True)

plt.tight_layout()
plt.show()


# ======================================================================
# ## 4. Interpretation
# ======================================================================

print("\n--- Coefficient Interpretation ---")
for feature, coef in zip(X.columns, model.coef_):
    if coef > 0:
        print(f"• {feature}: A 1-unit increase leads to a €{coef:.2f} increase in price (holding others constant)")
    else:
        print(f"• {feature}: A 1-unit increase leads to a €{abs(coef):.2f} decrease in price (holding others constant)")


most_influential_idx = np.argmax(np.abs(model.coef_))
most_influential_feature = X.columns[most_influential_idx]
most_influential_coef = model.coef_[most_influential_idx]

print(f"\n--- Most Influential Feature ---")
print(f"Feature: {most_influential_feature}")
print(f"Coefficient: {most_influential_coef:.2f}")
print(f"\nThis feature has the largest absolute impact on housing price predictions.")


plt.figure(figsize=(10, 6))
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient')

plt.barh(coefficients['Feature'], coefficients['Coefficient'])
plt.xlabel('Coefficient Value')
plt.title('Feature Coefficients (Impact on Price)')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

