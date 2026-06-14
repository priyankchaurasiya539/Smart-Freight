import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

#Load the artifacts
artifacts = joblib.load("models/processed_data_pipeline.pkl")
X_train = artifacts["X_train_encoded"]
X_test = artifacts["X_test_encoded"]

#Load the raw data
df_raw = pd.read_csv("data/DataCoSupplyChainDataset.csv", encoding="latin1")
y_raw = df_raw["Late_delivery_risk"]

#Same split for targets matching
m, n, y_train, y_test = train_test_split(y_raw, y_raw, test_size=0.25, random_state=42)

#Train XGBoost
xgb_clf = XGBClassifier(n_estimators = 300 , max_depth = 9 , learning_rate = 0.1 , subsample = 0.9 ,   colsample_bytree = 0.9 , random_state=42 , n_jobs = -1)
xgb_clf.fit(X_train, y_train)

# 5. Output Results
y_pred = xgb_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy score is:" , acc)

#Save the model
joblib.dump(xgb_clf, "models/final_xgb_model.pkl")
print("File Saved Successfully.")