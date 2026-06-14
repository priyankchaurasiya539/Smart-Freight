import pandas as pd 
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder , TargetEncoder

#Load the dataset 
#To build a crash-proof data ingestion pipeline, I explicitly configured the pandas engine with latin1 encoding, which handles single-byte mappings byte-by-byte without breaking production execution."

df = pd.read_csv("data/DataCoSupplyChainDataset.csv" , encoding='latin1')

print(df.head(10))
print("-" * 100)

#olumns
print(df.columns)
print("-" * 100)

#Shape
print("Shape : " , df.shape)
print("-" * 100)

#Check missing values
print(df.isnull().sum())
print("-" * 100)

#Dropping unncessary columns from the dataset
df_new = df.drop(columns = ["Category Id" , "Customer Fname"  , "Customer Id" , "Customer Lname" , "Customer Password" , "Customer Zipcode" , "Department Id" , "Latitude", "Longitude" , "Order Customer Id" ,"Order Id" ,"Order Item Cardprod Id" , "Order Item Id" ,"Order Zipcode" , "Product Card Id" , "Product Category Id" , "Product Description" , "Product Image" , "Customer Email" , "Customer Street" , "Order Status" ,  "Product Status","Customer Country" , "Customer State" ,"Delivery Status" , "Order Country" , "Order Region" , "Order State" , "Product Name"])
print(df_new.head(10))
print("-" * 100)

print("New Data shape : " , df_new.shape)
print("-" * 100)

print("New columns :\n" , df_new.columns)
print("-" * 100)

print(df['Order City'].nunique())

print(df_new.info())

np.random.seed(42)  #To assign the probability mapping outputs and repeat them and to assign a static value 

#Origin City Weather Simulation (Day 0 - Current Status)
weather_conditions = ["Clear", "Rainy", "Foggy", "Heavy_Snow"]
df_new["Origin_Weather_Current"] = np.random.choice(
    weather_conditions, size=len(df_new), p=[0.5, 0.2, 0.2, 0.1]
)   
#Destination city weather 
df_new["Dest_Weather_Risk_Score"] = np.random.uniform(0.1, 0.9, size=len(df_new))   #Value more nearer to 0.1 signifies weather of destination city is clear and if it is more nearer to 0.9 then it signifies the weather is not clear

X = df_new.drop(columns=["Late_delivery_risk" , "Days for shipping (real)"])
y_clf = df_new["Late_delivery_risk"]
y_reg = df_new["Days for shipping (real)"]

X_train , X_test , y_train_clf , y_test_clf , y_train_reg , y_test_reg = train_test_split(X, y_clf , y_reg , random_state=42 , test_size=0.25)


#Data encoding 

#Category Name

target_enc_category_name = TargetEncoder(smooth="auto" , random_state=42)
X_train["Category Name"] = target_enc_category_name.fit_transform(X_train[["Category Name"]] , y_train_clf)
X_test["Category Name"] = target_enc_category_name.transform(X_test[["Category Name"]])


#Type(OHE)


ohe_cols = [
    "Type",
    "Customer Segment",
    "Department Name",
    "Market",
    "Origin_Weather_Current",
    "Shipping Mode",
]

X_train = pd.get_dummies(X_train, columns=ohe_cols, drop_first=True, dtype=float)
X_test = pd.get_dummies(X_test, columns=ohe_cols, drop_first=True, dtype=float)

#drop_first = true ( it drops the first value of categorical data by alphabetically order )     

#Order City 
target_enc_city = TargetEncoder(smooth="auto" , random_state=42)
X_train["Order City"] = target_enc_city.fit_transform(X_train[["Order City"]] , y_train_clf)
X_test["Order City"] = target_enc_city.transform(X_test[["Order City"]])

X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0.0)


#saving the data
pipeline_artifacts = {
    "X_train_encoded" : X_train ,
    "X_test_encoded" : X_test,
    "target_enc_city" : target_enc_city,
    "target_enc_category_name" : target_enc_category_name

}

joblib.dump(pipeline_artifacts , "models/processed_data_pipeline.pkl")
print("All files saved successfully.")