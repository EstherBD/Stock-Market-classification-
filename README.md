# Stock Market (classification)

Research notebook for S&P 500 classification, built using the S&P 500 dataset (version 984, ~202 MB) by andrewmvd. The project cleans and merges the three core files (`sp500_companies.csv`, `sp500_index.csv`, `sp500_stocks.csv`), performs data exploration and feature engineering (returns, rolling statistics, volatility, sector metadata), and trains machine learning models (Logistic Regression, Random Forest, XGBoost, HistGradientBoosting) to predict next-week stock movements. The notebook provides reproducible steps, detailed notes, and evaluation using metrics such as Accuracy, ROC-AUC, and F1, with feature selection guided by permutation importance. It also includes clear plots and explanations, making this a practical starting template for financial data classification tasks.

## Dataset

Source: **Kaggle — S&P 500 Stocks (version 984, ~202 MB)**  
Link: https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks/versions/984
Files used:
- `sp500_companies.csv`
- `sp500_index.csv`
- `sp500_stocks.csv`


## Project Workflow

### 1. Data Loading and Preprocessing
- Combined three Kaggle datasets
- Normalized **tickers** and **sectors**.
- Handled missing values in OHLCV data using forward/backward fills.
- Corrected anomalies:
  - Zero trading volume on weekdays: set to NaN and re-filled.
  - Enforced **price envelope** rule: `Low ≤ {Open, Close, Adj Close} ≤ High`.
- Created a cleaned, merged dataset with index values and sector metadata.

### 2. Exploratory Data Analysis (EDA)
- **Index trend**: plotted S&P 500 index over time.
- **Sector distribution**: bar chart of stock counts per sector.
- **Performance analysis**:
  - Top and bottom stocks by total return.
  - Sector-level performance (average and median % returns).
- **Volatility**:
  - Median daily-return volatility per sector.
- **Correlations**:
  - Heatmap of sector daily-return correlations.
  - Pairwise scatter of daily returns (e.g., AAPL vs MSFT).
- **Histograms** of daily % returns for individual tickers.

### 3. Feature Engineering
- **Per-stock features**:
  - Lagged returns (1, 2, 3 days).
  - Lagged volume.
  - Moving averages (5-day, 20-day).
  - Rolling volatility (5-day, 20-day).
  - Cumulative 3-day return.
  - Up-days in last 5 sessions.
- **Calendar features**:
  - Day of week.
  - Month.
- **Market features**:
  - Lagged market return (S&P 500).
- **Target**:
  - **`Up_in_5`** : binary label indicating if price goes up after 5 days.

### 4. Modeling
- **Baselines**:
  - Always-majority class.
  - 5-day momentum rule.
- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **HistGradientBoosting**
- Hyperparameter tuning with **TimeSeriesSplit + RandomizedSearchCV**.
- Coefficient inspection for Linear Regression, feature importances for tree-based models, and permutation importances for HistGradientBoosting.
    

### 5. Evaluation
- **Metrics**:
  - Accuracy.
  - ROC-AUC.
  - Precision-Recall AUC.
- **Plots**:
  - Confusion matrices for each model.
  - ROC curves and PR curves overlayed.
- **Permutation importance**:
  - Identified top drivers for the best model (Logistic Regression).


### 6. Feature Reduction and High-Confidence Cases
- Selected reduced feature set (`MA_20`, `Month`, `UpDays_5`, `lag_1_mrkt_return`, `lag_1_volume`).
- Refit Logistic Regression on reduced features with tuned hyperparameters.
- Compared performance with full model.
- Inspected **high-confidence correct predictions** vs **confident mistakes** to analyze edge cases.


## Results
- Baseline models set a simple reference (majority, momentum).
- Logistic Regression outperformed all models and provided interpretable coefficients.
- Feature reduction simplified the model while retaining competitive accuracy.

