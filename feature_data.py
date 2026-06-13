import pandas as pd 

#Load the dataset 
#To build a crash-proof data ingestion pipeline, I explicitly configured the pandas engine with latin1 encoding, which handles single-byte mappings byte-by-byte without breaking production execution."

df = pd.read_csv("DataCoSupplyChainDataset.csv" , encoding='latin1')

print(df.head(10))
print("-" * 100)

#Columns
print(df.columns)
print("-" * 100)

#Shape
print("Shape : " , df.shape)
print("-" * 100)

#Check missing values
print(df.isnull().sum())
print("-" * 100)

#Dropping unncessary columns from the dataset
df_new = df.drop(columns = ["Category Id" , "Customer Fname"  , "Customer Id" , "Customer Lname" , "Customer Password" , "Customer Zipcode" , "Department Id" , "Latitude", "Longitude" , "Order Customer Id" ,"Order Id" ,"Order Item Cardprod Id" , "Order Item Id" ,"Order Zipcode" , "Product Card Id" , "Product Category Id" , "Product Description" , "Product Image" , "Customer Email" , "Customer Street" , "Order Status" ,  "Product Status" , "Customer Country"])
print(df_new.head(10))
print("-" * 100)

print("New Data shape : " , df_new.shape)
print("-" * 100)

print("New columns :\n" , df_new.columns)
print("-" * 100)

print(df['Order City'].nunique())

print(df_new.info())