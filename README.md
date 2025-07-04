# Laptop Price Prediction

This repository contains a Jupyter Notebook (`LaptopPricePrediction.ipynb`) that outlines a machine learning project focused on predicting laptop prices. The notebook details the entire process from data loading and extensive preprocessing to feature engineering, model training, and evaluation.

## Table of Contents

  - [Project Overview](https://github.com/nouninasion/LaptopPricePrediction/blob/main/README.md#project-overview)
  - [Dataset](https://github.com/nouninasion/LaptopPricePrediction/blob/main/README.md#dataset)
  - [Features](https://github.com/nouninasion/LaptopPricePrediction/blob/main/README.md#features)
  - [Preprocessing and Feature Engineering](https://github.com/nouninasion/LaptopPricePrediction/blob/main/README.md#preprocessing-and-feature-engineering)
  - [Exploratory Data Analysis (EDA)](https://github.com/nouninasion/LaptopPricePrediction/blob/main/README.md#exploratory-data-analysis-eda)
  - [Model Training](https://github.com/nouninasion/LaptopPricePrediction/blob/main/README.md#model-training)
  - [Model Performance](https://github.com/nouninasion/LaptopPricePrediction/blob/main/README.md#model-performance)
  - [Prediction Example](https://github.com/nouninasion/LaptopPricePrediction/blob/main/README.md#prediction-example)
  - [Getting Started](https://github.com/nouninasion/LaptopPricePrediction/blob/main/README.md#getting-started)
  - [Libraries Used](https://github.com/nouninasion/LaptopPricePrediction/blob/main/README.md#libraries-used)

## Project Overview

The main objective of this project is to develop a robust machine learning model for predicting laptop prices. The notebook demonstrates a comprehensive data science pipeline, including advanced data cleaning, transformation, and the application of various regression algorithms to achieve accurate price estimations.

## Dataset

The dataset used for this project is `laptop_price.csv`. It contains a wide range of laptop specifications and their corresponding prices in euros.

## Features

The dataset initially includes the following features:

  - `laptop_ID`: Unique identifier for each laptop.
  - `Company`: Manufacturer of the laptop.
  - `Product`: Product name.
  - `TypeName`: Type of laptop (e.g., Notebook, Gaming).
  - `Inches`: Screen size in inches.
  - `ScreenResolution`: Resolution and screen type (e.g., 'IPS Panel Full HD 1920x1080').
  - `Cpu`: CPU specifications (e.g., 'Intel Core i5 2.3GHz').
  - `Ram`: RAM capacity (e.g., '8GB').
  - `Memory`: Storage specifications (e.g., '1TB HDD', '256GB SSD').
  - `Gpu`: GPU specifications (e.g., 'Intel HD Graphics 620').
  - `OpSys`: Operating system.
  - `Weight`: Weight of the laptop (e.g., '1.37kg').
  - `Price_euros`: The price of the laptop in euros (target variable).

## Preprocessing and Feature Engineering

Extensive steps were performed to clean, transform, and create new features from the raw data:

  - **Handling Missing Values**: Missing values in numerical features were imputed using the median, while categorical features had their missing values filled with 'Others' or the mode.
  - **Duplicate Removal**: Duplicate rows were identified and removed.
  - **Feature Cleaning and Extraction**:
      - `Weight`: 'kg' suffix was removed, and the column was converted to float.
      - `Ram`: 'GB' suffix was removed, and the column was converted to integer.
      - `Cpu`: CPU speed (GHz) was extracted. 'Intel Core' was simplified.
      - `Gpu`: GPU manufacturer was extracted.
      - `Memory`: This complex column was parsed to extract separate numerical columns for `SSD` and `HDD` storage (in GB), and other storage types like 'Flash Storage' and 'Hybrid' were handled separately.
      - `ScreenResolution`: X and Y resolutions (`X_res`, `Y_res`) were extracted, and a `PPI` (Pixels Per Inch) feature was calculated. Boolean features like `IPS_Panel` and `Touchscreen` were also extracted.
  - **Feature Encoding**: Categorical features including `Company`, `Product`, `TypeName`, `Cpu` (after simplification), `Ram`, `Gpu` (after extraction), `OpSys`, and `Weight` were transformed into numerical representations using `LabelEncoder`.
  - **Log Transformation**: The `Price_euros` (target variable) was log-transformed (`np.log`) to normalize its distribution.
  - **Dropping Original Columns**: Original raw columns that were processed into new features were dropped to prevent redundancy and simplify the dataset.

## Exploratory Data Analysis (EDA)

The notebook includes a comprehensive EDA section:

  - Initial data inspection with `data.info()` and `data.describe()` to understand data types, non-null counts, and descriptive statistics.
  - Histograms for numerical features and bar plots for categorical features to visualize their distributions.
  - A heatmap of the correlation matrix between features was generated to identify relationships and potential multicollinearity.

## Model Training

The processed data was split into training and testing sets. The following regression models were trained:

  - Linear Regression
  - RandomForestRegressor
  - XGBoost (XGBRegressor)

## Model Performance

The performance of the models was evaluated using the R-squared score on the test set:

  - **Linear Regression**: R-squared score: `0.7415559521793839`.
  - **RandomForestRegressor**: R-squared score: `0.7789430494759759`.
  - **XGBoost**: R-squared score: `0.8355623437162947`.

XGBoost provided the slightly best performance, closely followed by RandomForestRegressor, indicating their effectiveness in predicting laptop prices.

## Prediction Example

The notebook includes a section demonstrating how to use the best-trained model (XGBoost) to predict the price of a new, hypothetical laptop based on its specifications.

## Getting Started

To run this notebook, you will need Google Colab or a local Jupyter environment with the following libraries installed:

  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `xgboost`
  - `tabulate`

You can install these using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tabulate
```

Once the dependencies are installed, you can open and run the `LaptopPricePrediction.ipynb` file in your Google Colab or Jupyter environment.

## Libraries Used

  - `pandas`
  - `numpy`
  - `matplotlib.pyplot`
  - `seaborn`
  - `sklearn.model_selection`
  - `sklearn.preprocessing`
  - `sklearn.impute`
  - `sklearn.linear_model`
  - `sklearn.ensemble`
  - `xgboost`
  - `re` (Python's built-in regular expression module)
  - `tabulate`
