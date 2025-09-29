# Stock Market (classification)

Research notebook for S&P 500 classification, built using the S&P 500 dataset (version 984, ~202 MB) by andrewmvd. The project cleans and merges the three core files (`sp500_companies.csv`, `sp500_index.csv`, `sp500_stocks.csv`), performs data exploration and feature engineering (returns, rolling statistics, volatility, sector metadata), and trains machine learning models (Logistic Regression, Random Forest, XGBoost, HistGradientBoosting) to predict next-week stock movements. The notebook provides reproducible steps, detailed notes, and evaluation using metrics such as Accuracy, ROC-AUC, and F1, with feature selection guided by permutation importance. It also includes clear plots and explanations, making this a practical starting template for financial data classification tasks.


## Dataset

Source: **Kaggle — S&P 500 Stocks (version 984, ~202 MB)**  
Link: https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks/versions/984
Files used:
- `sp500_companies.csv`
- `sp500_index.csv`
- `sp500_stocks.csv`

