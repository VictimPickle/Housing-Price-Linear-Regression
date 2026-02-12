# Housing Price Prediction with Linear Regression

## Objective

Build, evaluate, and interpret a linear regression model to predict housing prices in a city based on property features. This project practices data splitting, model training, evaluation, and interpretation while adhering to machine learning best practices.

## Dataset

The dataset contains housing data with the following features:

### Features
- **area_m2**: Area in square meters
- **bedrooms**: Number of bedrooms
- **distance_km**: Distance to city center in kilometers
- **age_years**: Age of property in years

### Target Variable
- **price**: Housing price (target to predict)

## Project Tasks

### 1. Data Exploration
Load and inspect the dataset structure using:
- `head()`: View first few rows
- `describe()`: Statistical summary of features
- Data visualization and distribution analysis

### 2. Model Training (30%)
Train a linear regression model using:
- **Algorithm**: `sklearn.linear_model.LinearRegression`
- **Training set**: Only train on designated training data
- **Output**: Model coefficients and intercept

The model learns the relationship between features and price:
```
price = β₀ + β₁·area + β₂·bedrooms + β₃·distance + β₄·age
```

### 3. Model Evaluation (20%)
Predict prices for both training and test sets, then calculate:

- **MSE (Mean Squared Error)**: Average squared difference between predicted and actual prices
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in same units as price
- **R² Score**: Proportion of variance explained by the model (0 to 1)

Compare training vs. test performance to detect overfitting or underfitting.

### 4. Interpretation (20%)

#### Coefficient Analysis
Explain the meaning of each coefficient:
- Example: "A 1 km increase in distance reduces price by €X, holding other features constant"
- Identify positive vs. negative relationships
- Determine statistical significance

#### Feature Importance
Identify the most influential feature based on:
- Magnitude of coefficients (after standardization)
- Impact on price predictions
- Real-world interpretation

## Project Structure

```
Housing-Price-Linear-Regression/
│
├── housing_data.csv              # Dataset with features and prices
├── housing_price_prediction.py   # Python implementation
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Requirements

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/VictimPickle/Housing-Price-Linear-Regression.git
   cd Housing-Price-Linear-Regression
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Python script:
   ```bash
   python housing_price_prediction.py
   ```

4. The script will:
   - Load and explore the data
   - Train the linear regression model
   - Display coefficients and intercept
   - Calculate evaluation metrics (MSE, RMSE, R²)
   - Print interpretations of coefficients
   - Identify the most influential feature

## Expected Output

The program will display:
- Dataset statistics and distributions
- Model coefficients for each feature
- Training and test set performance metrics
- Interpretation of each coefficient's impact on price
- Most influential feature affecting housing prices

## Learning Outcomes

✅ Understanding linear regression fundamentals

✅ Proper train-test data splitting

✅ Model evaluation with multiple metrics

✅ Coefficient interpretation and feature importance

✅ Machine learning best practices

## Technologies Used

- **Python 3.x**
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Linear regression implementation
- **Matplotlib/Seaborn**: Data visualization

## Author

Mobin Ghorbani  
CS Student  
Machine Learning Project

## License

This project is for educational purposes.
