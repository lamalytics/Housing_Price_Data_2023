import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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

sns.histplot(data=train_set, x="SalePrice", color="orange")
sns.displot(data=train_set, x="MSZoning", color="blue")


corr_matrix = train_set.corr(method='spearman', numeric_only=True)
sns.heatmap(corr_matrix, cmap='coolwarm')
plt.show()

# print(train_set.isnull().sum().sort_values(ascending=False))
# print(test_set.isnull().sum().sort_values(ascending=False))

# some are categorical; some numerical
# imputer = SimpleImputer(strategy='mean')  # you can also use median or mode
train_set.drop(columns=["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "Id"], inplace=True)
# print(train_set.isnull().sum().sort_values(ascending=False))

# scaling categorical variables
# NEED to scale numerical values to normalize
features = train_set.drop(columns=["SalePrice"])
target = np.array(train_set.loc[:,["SalePrice"]])

# select numerical cols and categorical for scaling
numerical_cols = features.select_dtypes(include=["float64", "int64"]).columns
categorical_cols = features.select_dtypes(include=["object"])

# scaling numerical and replace numerical cols with scaled nums
min_max_scaler = MinMaxScaler()
features[numerical_cols] = min_max_scaler.fit_transform(features[numerical_cols].values)
train_set_encoded = pd.get_dummies(features, columns=categorical_cols.columns)
# combine the scaled numerical features and dummy encoded categorical features
final_features = np.concatenate((features[numerical_cols].values, train_set_encoded.values), axis=1)

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

# next step hypertuning
xgb_model = xgb.XGBRegressor()
xgb_model.fit(train_features, train_target)
predict_target = xgb_model.predict(test_features)

rmse = MSE(test_target, predict_target) ** (1/2)
print(rmse)
# RMSE at 33K :(

