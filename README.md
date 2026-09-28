# E-commerce Customer Spending Prediction

A beginner-friendly **Machine Learning regression project** that predicts a customer's **Yearly Amount Spent** using e-commerce customer behavior data.

The project follows a standard machine learning workflow:

**Data Understanding → EDA → Preprocessing → Model Training → Prediction → Evaluation → Error Analysis**

---

## Project Overview

E-commerce companies collect information about how customers interact with their platforms.

In this project, we use customer behavior features such as:

- Average Session Length
- Time spent on the App
- Time spent on the Website
- Length of Membership

to predict:

> **Yearly Amount Spent**

The primary model used in this project is **Multiple Linear Regression**.

---



## Problem Statement

Given information about an e-commerce customer's usage behavior and membership duration, can we predict how much they will spend annually?

### Target Variable

`Yearly Amount Spent`

### Input Features


| Feature                | Description                              |
| ---------------------- | ---------------------------------------- |
| `Avg. Session Length`  | Average duration of a customer's session |
| `Time on App`          | Time spent using the mobile application  |
| `Time on Website`      | Time spent using the website             |
| `Length of Membership` | Duration of the customer's membership    |


---



## Dataset

The dataset contains **500 customer records** and **8 columns**.

### Columns

```text
Email
Address
Avatar
Avg. Session Length
Time on App
Time on Website
Length of Membership
Yearly Amount Spent
```

For the initial model, only the numerical behavioral features are used.

The following columns are excluded:

- `Email`
- `Address`
- `Avatar`

These columns are not used as predictors in this project because they are identifiers or non-numeric information and are not part of the initial regression modeling approach.

---



## Machine Learning Approach

The project uses **Multiple Linear Regression**.

The general equation is:

ŷ = β₀ + β₁X₁ + β₂X₂ + β₃X₃ + β₄X₄

where:

- `ŷ` = predicted yearly amount spent
- `β₀` = intercept
- `β₁ ... β₄` = model coefficients
- `X₁ ... X₄` = input features

The model estimates the coefficients that minimize the prediction error on the training data.

---



# Project Workflow



## 1. Exploratory Data Analysis

The first stage is understanding the dataset before training a model.

The EDA includes:

- Dataset structure
- Number of rows and columns
- Data types
- Summary statistics
- Missing-value analysis
- Duplicate-value analysis
- Unique-value analysis
- Distribution of numerical variables
- Outlier inspection
- Correlation analysis
- Feature-target relationships
- Pairwise relationships



### Visualizations

The project uses:

- Histograms
- Boxplots
- Scatterplots
- Regression plots
- Correlation heatmap
- Pairplot

The purpose of EDA is to understand the data and identify patterns before modeling.

---



## 2. Data Preprocessing

The selected features and target variable are separated:

```python
X = df[features]
y = df[target]
```

The dataset is then divided into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```



### Train/Test Split

- **80%** → Training data
- **20%** → Testing data

The test set is kept separate so that the model can be evaluated on data it did not see during training.

### Why `random_state=42`?

`random_state` makes the split reproducible.

The value `42` is simply a commonly used fixed seed. It has no special mathematical meaning.

---



## 3. Baseline Model

Before evaluating Linear Regression, a simple baseline is created.

The baseline predicts the **mean training target value for every test observation**.

This provides a reference point to determine whether the trained model actually provides useful predictive performance.

A machine learning model should generally be compared against a reasonable baseline rather than evaluated in isolation.

---



## 4. Linear Regression Model

The model is created using Scikit-learn:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

The model learns:

- Intercept
- Feature coefficients

The learned coefficients can be inspected to understand the direction and magnitude of the model's linear relationships.

### Coefficient Interpretation

For a coefficient:

> Holding the other included predictors constant, a one-unit increase in that feature is associated with a change in the predicted target equal to the coefficient.

These coefficients describe **association within the fitted model** and should not automatically be interpreted as causal effects.

---



# Model Evaluation

The model is evaluated using multiple regression metrics.

## Mean Absolute Error (MAE)

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$

MAE represents the average absolute prediction error.

It is expressed in the same units as the target variable.

---



## Mean Squared Error (MSE)

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

MSE squares the errors before averaging them.

Therefore, larger errors receive greater penalty.

---



## Root Mean Squared Error (RMSE)

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

RMSE is expressed in the same units as the target and is more sensitive to large errors than MAE.

---



## R² Score

$$
R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
{\sum_{i=1}^{n}(y_i - \bar{y})^2}
$$

R² measures how much of the variation in the target is explained by the model relative to a mean-prediction baseline on the evaluated dataset.

**Important:** R² should not be interpreted as model "accuracy."

---



# Error Analysis

Model evaluation does not stop at calculating metrics.

The project also investigates the model's prediction errors.

### Actual vs Predicted

The actual and predicted values are plotted against each other.

A 45° reference line represents:

```text
Actual = Predicted
```

Points closer to this line indicate smaller prediction errors.

---



### Residual Distribution

Residual:

Residual = Actual - Predicted

The residual distribution is inspected to determine whether errors are approximately centered around zero and whether there are unusual patterns or large errors.

---



### Residuals vs Predicted Values

Residuals are plotted against predicted values to check for:

- Systematic patterns
- Curvature
- Changing variance
- Clusters
- Large unusual errors

Ideally, residuals should be scattered around zero without a strong systematic pattern.

---



# Project Structure

```text
ecommerce-linear-regression/
│
├── data/
│   └── Ecommerce Customers.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Linear_Regression.ipynb
│   └── 04_Model_Evaluation.ipynb
│
├── models/
│
├── reports/
│
├── .gitignore
├── requirements.txt
└── README.md
```

---



# Technologies Used



### Programming Language

- Python



### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn



### Environment

- Jupyter Notebook



### Version Control

- Git
- GitHub

---



# Installation



## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ecommerce-linear-regression.git
```

Move into the project directory:

```bash
cd ecommerce-linear-regression
```

---



## 2. Create a virtual environment

```bash
python -m venv .venv
```



### Windows

```bash
.venv\Scripts\activate
```



### macOS / Linux

```bash
source .venv/bin/activate
```

---



## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---



## 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Then open the notebooks in the following order:

```text
01_EDA.ipynb
        ↓
02_Preprocessing.ipynb
        ↓
03_Linear_Regression.ipynb
        ↓
04_Model_Evaluation.ipynb
```

---



# Key Learning Outcomes

Through this project, I learned how to:

- Understand a real-world dataset
- Perform exploratory data analysis
- Identify numerical and categorical columns
- Analyze missing values and duplicates
- Understand feature distributions
- Analyze correlations
- Separate features and target variables
- Split data into training and testing sets
- Understand data leakage
- Establish a baseline model
- Train a Multiple Linear Regression model
- Interpret regression coefficients
- Generate predictions
- Calculate MAE, MSE, RMSE and R²
- Compare a model against a baseline
- Analyze residuals
- Interpret Actual vs Predicted plots
- Analyze model errors
- Structure an ML project using Git and GitHub

---



# Important ML Concepts Demonstrated

This project focuses on understanding the fundamentals rather than simply obtaining a high metric.

### Concepts covered

```text
Exploratory Data Analysis
        ↓
Train/Test Split
        ↓
Baseline
        ↓
Multiple Linear Regression
        ↓
Predictions
        ↓
Regression Metrics
        ↓
Residual Analysis
        ↓
Model Interpretation
```

---



# Limitations

This is an introductory Linear Regression project.

The project does **not** currently include:

- Feature engineering
- Hyperparameter tuning
- Cross-validation
- Regularization
- Advanced ensemble models
- Deployment
- Experiment tracking
- Production ML pipelines

These topics can be introduced in later projects as the machine learning workflow becomes more advanced.

---



# Future Improvements

Possible future extensions include:

- Ridge Regression
- Lasso Regression
- Cross-validation
- Hyperparameter tuning
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- Model serialization
- API development
- Model deployment
- Experiment tracking
- ML pipelines

---



# Conclusion

This project demonstrates a complete beginner-level regression workflow, from understanding the dataset and performing EDA to training, evaluating, and analyzing a Linear Regression model.

The main goal is not only to build a prediction model, but also to understand **why each step is performed and how to interpret the resulting model and errors**.

---

