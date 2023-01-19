import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

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


pre_train_set, post_train_set = train_test_split(train_set, test_size=0.2, random_state=21)
pre_train_set = pd.DataFrame(pre_train_set)

features = pre_train_set.drop(columns=["SalePrice"])
target = pre_train_set.loc[:,["SalePrice"]]

encoder = OneHotEncoder(handle_unknown='ignore')
features_encoded = encoder.fit_transform(features.select_dtypes(include=["object"]))
# print(features_encoded)
features_numeric = features.select_dtypes(exclude=["object"])
# print(features_numeric)

final_features = np.concatenate(features_encoded.toarray(), features_numeric, axis=1)
xgb = XGBRegressor(enable_categorical=True)
# # scaler on this
# xgb.fit(features,target)

# pred_target = xgb.predict(features)

# print(pred_target)

# test_set.drop(columns=["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu"], inplace=True)
# print(test_set.isnull().sum().sort_values(ascending=False))
# print(test_set.head())



# # Fit the imputer on the data
# imputer.fit(train_set)

# # Apply the imputer to the data
# df_imputed = imputer.transform(train_set)
