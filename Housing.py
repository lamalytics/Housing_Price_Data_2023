import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error as MSE

train_set = pd.read_csv(
    "house-prices-advanced-regression-techniques-data/train.csv")
test_set = pd.read_csv(
    "house-prices-advanced-regression-techniques-data/test.csv")

# EDA
# print(train_set.tail())
# print(test_set.tail())

# print(train_set.shape)
# print(test_set.shape)

# print(train_set.describe())
# print(test_set.describe())

# print(train_set.isnull().sum().sort_values(ascending=False))
# print(test_set.isnull().sum().sort_values(ascending=False))

# some are categorical; some numerical
# imputer = SimpleImputer(strategy='mean')  # you can also use median or mode
train_set.drop(columns=["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"], inplace=True)
# print(train_set.isnull().sum().sort_values(ascending=False))

# scaling categorical variables
# NEED to scale numerical values to normalize
features = train_set.drop(columns=["SalePrice"])
target = np.array(train_set.loc[:,["SalePrice"]])
categorical_cols = features.select_dtypes(include=["object"])
train_set_encoded = pd.get_dummies(features, columns=categorical_cols.columns)
final_features = np.array(train_set_encoded)

# partitition training set since test_set has no sales price for prediction
train_features, test_features, train_target, test_target = train_test_split(final_features,target, test_size=0.2, random_state=21)

# print(train_target)

# select only categorical columns
# categorical_cols = features.select_dtypes(include=["object"]).columns
# test_cat_cols = test_features.select_dtypes(include=["object"]).columns
# encoding cat variables as dummies on pre_train set
# train_set_encoded = pd.get_dummies(features, columns=categorical_cols)
# test_set_encoded = pd.get_dummies(test_features, columns=test_cat_cols)
# features_final = train_set_encoded.values

xgb_model = xgb.XGBRegressor()
# make target from DF to a 1D array 
# xgb_model.fit(features_final, target.values.ravel())
xgb_model.fit(train_features, train_target)
predict_target = xgb_model.predict(test_features)

rmse = MSE(test_target, predict_target) ** (1/2)
print(rmse)
# RMSE at 33K :(

