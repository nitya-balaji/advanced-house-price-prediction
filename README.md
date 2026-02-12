# House Price Prediction Model: End-to-End ML Pipeline

A comprehensive Machine Learning pipeline for predicting house prices using the Ames Housing dataset. This project implements advanced data preprocessing, feature engineering, and model training techniques to achieve accurate price predictions.

## Project Overview

This project demonstrates a complete end-to-end Machine Learning workflow, from initial data exploration to final predictions. The model predicts house sale prices based on 80 explanatory variables describing various aspects of residential homes in Ames, Iowa.

**Final Model Performance:** Random Forest Regressor with 21 carefully selected features from the 82 total variables in the dataset.

## Project Structure

```
house-price-prediction/
│
├── eda.ipynb                          # Exploratory Data Analysis
├── feature_engineering.ipynb          # Data preprocessing and feature engineering
├── feature_selection.ipynb            # Feature selection using Lasso
├── house_prices_prediction.ipynb      # Model training and predictions
│
├── train.csv                          # Original training dataset
├── test.csv                           # Original test dataset
├── X_train.csv                        # Processed training data 
├── X_test.csv                         # Processed test data 
└── submission.csv                     # Final predictions
```

## Workflow

### 1. Exploratory Data Analysis (EDA)

**File:** `eda.ipynb`

Comprehensive analysis of the dataset to understand:
- Dataset shape and structure (1,460 rows × 82 columns)
- Distribution of numerical and categorical features
- Missing value patterns and percentages
- Correlation analysis between features and target variable (SalePrice)
- Visualization of relationships between categorical features and sale prices using bar plots

**Key Insights:**
- Identified temporal features (YearBuilt, YearRemodAdd, GarageYrBlt)
- Discovered skewed numerical distributions requiring transformation
- Found significant missing values in features like LotFrontage, MasVnrArea, and GarageYrBlt
- Analyzed categorical features and their impact on sale price

### 2. Feature Engineering

**File:** `feature_engineering.ipynb`

Applied advanced preprocessing techniques to both training and test datasets:

#### Missing Value Handling
- **Categorical features:** Replaced NaN values with "Missing" category
- **Numerical features:** Filled with median values and created binary indicator columns (e.g., `LotFrontagenan`, `MasVnrAreanan`, `GarageYrBltnan`) to preserve information about missingness

#### Feature Transformations
- **Temporal Variables:** Converted absolute years to relative age
  - `YearBuilt` → Years since sale
  - `YearRemodAdd` → Years since remodel
  - `GarageYrBlt` → Garage age at sale
  
- **Log Transformation:** Applied to skewed numerical features to normalize distributions
  - `LotFrontage`, `LotArea`, `1stFlrSF`, `GrLivArea`

#### Categorical Feature Engineering
- **Rare Category Handling:** Grouped categories appearing in <1% of data as "Rare_var"
- **Target-Guided Ordinal Encoding:** Encoded categorical variables based on their mean SalePrice
  - Preserves ordinal relationship with target variable
  - More informative than one-hot encoding for tree-based models

#### Feature Scaling
- Applied MinMaxScaler to normalize all features to [0, 1] range
- Ensures consistent scale across all features for model training

### 3. Feature Selection

**File:** `feature_selection.ipynb`

Used Lasso Regression (L1 regularization) to identify the most predictive features:

**Process:**
- Lasso automatically performs feature selection by shrinking less important feature coefficients to zero
- Used `SelectFromModel` to extract features with non-zero coefficients
- Reduced feature space from 82 to **21 essential features**

**Selected Features:**
1. MSSubClass
2. MSZoning
3. Neighborhood
4. OverallQual
5. YearRemodAdd
6. RoofStyle
7. BsmtQual
8. BsmtExposure
9. HeatingQC
10. CentralAir
11. 1stFlrSF
12. GrLivArea
13. BsmtFullBath
14. KitchenQual
15. Fireplaces
16. FireplaceQu
17. GarageType
18. GarageFinish
19. GarageCars
20. PavedDrive
21. SaleCondition

### 4. Model Training & Prediction
**File:** `house_prices_prediction.ipynb`

#### Model: Random Forest Regressor
- **Algorithm:** Ensemble of 100 decision trees
- **Random State:** 42 (for reproducibility)

#### Training Process
1. Loaded processed datasets (`X_train.csv`, `X_test.csv`)
2. Separated features (21 selected features) and target (SalePrice)
3. Trained Random Forest on engineered features
4. Generated predictions on test set

#### Prediction Pipeline
```python
log_predictions = model.predict(X_test)
final_predictions = np.exp(log_predictions) #Reverse log transformation
```

**Important:** Since SalePrice was log-transformed during feature engineering, predictions are exponentiated to return to original scale.

## Technologies Used

- **Python 3.10.11**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **seaborn** - Statistical visualizations
- **scikit-learn** - Machine learning algorithms and preprocessing
  - `RandomForestRegressor` - Final model
  - `Lasso` - Feature selection
  - `MinMaxScaler` - Feature scaling
  - `SelectFromModel` - Feature selector wrapper

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Ensure you have the dataset files:
   - `train.csv`
   - `test.csv`

## Usage
Run the notebooks in sequential order:

```bash
#1. Exploratory Data Analysis
jupyter notebook eda.ipynb

#2. Feature Engineering
jupyter notebook feature_engineering.ipynb

#3. Feature Selection
jupyter notebook feature_selection.ipynb

#4. Model Training and Prediction
jupyter notebook house_prices_prediction.ipynb
```
After running all notebooks, `submission.csv` will contain the final predictions ready for submission.

## Acknowledgments
**Dataset:** This project uses the **Ames Housing Dataset** from the Kaggle competition: [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
- **Kaggle Competition Score:** 0.29926 (RMSE) - A solid baseline demonstrating a functional end-to-end pipeline. Future improvements such as experimenting with different ML algorithms and advanced feature engineering could further reduce the RMSE score.
