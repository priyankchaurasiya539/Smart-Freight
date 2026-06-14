import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder

# 1. Dataset Load kiya
df = pd.read_csv("data/DataCoSupplyChainDataset.csv", encoding="latin1")
df_new = df.copy()

# Weather simulation logic 
np.random.seed(42)
weather_conditions = ["Clear", "Rainy", "Foggy", "Heavy_Snow"]
df_new["Origin_Weather_Current"] = np.random.choice(
    weather_conditions, size=len(df_new), p=[0.5, 0.2, 0.2, 0.1]
)
df_new["Dest_Weather_Risk_Score"] = np.random.uniform(0.1, 0.9, size=len(df_new))

#Remove the unncessary columns 

unwanted_cols = [
    "Late_delivery_risk", "Days for shipping (real)", "Delivery Status", 
    "Customer City", "Customer Country", "Customer Email", "Customer Fname", 
    "Customer Lname", "Customer Password", "Customer State", "Customer Street", 
    "Order Country", "order date (DateOrders)", "Order Region", "Order State", 
    "Order Status", "Product Image", "Product Name", "shipping date (DateOrders)"
]

X = df_new.drop(columns=unwanted_cols,  errors='ignore')
y_clf = df_new["Late_delivery_risk"]

#train_test_split 
X_train, X_test, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.25, random_state=42
)

#Target Encoding on Category Name
target_enc_category_name = TargetEncoder(smoothing=1.0)
X_train["Category Name"] = target_enc_category_name.fit_transform(X_train["Category Name"], y_train_clf)
X_test["Category Name"] = target_enc_category_name.transform(X_test["Category Name"])

#One Hot Encoding on simple categorical columns
ohe_cols = [
    "Type",
    "Customer Segment",
    "Department Name",
    "Market",
    "Origin_Weather_Current",
    "Shipping Mode"
]

X_train = pd.get_dummies(X_train, columns=ohe_cols, drop_first=True, dtype=float)
X_test = pd.get_dummies(X_test, columns=ohe_cols, drop_first=True, dtype=float)

#Target Encoding on Order City

target_enc_city = TargetEncoder(smoothing=1.0)
X_train["Order City"] = target_enc_city.fit_transform(X_train["Order City"], y_train_clf)
X_test["Order City"] = target_enc_city.transform(X_test["Order City"])

#Columns Align kiya 
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0.0)

#converting boolean columns to float
X_train = X_train.astype(float)
X_test = X_test.astype(float)

#Save artifacts
pipeline_artifacts = {
    "X_train_encoded": X_train,
    "X_test_encoded": X_test
}

joblib.dump(pipeline_artifacts, "models/processed_data_pipeline.pkl")
print("All files saved successfully and string noise dropped.")