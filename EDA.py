import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

#Load the dataset 
df = pd.read_csv("data/DataCoSupplyChainDataset.csv" , encoding="latin1")

# print(df.head(20))

"""Univariate Visualization"""


#Type 

plt.figure(figsize=(7, 6))
colors = ['cadetblue' , 'forestgreen' , 'crimson' , 'deeppink']
sns.countplot(data=df , x = 'Type', palette=colors , edgecolor = 'black')
plt.title("Distribution of Type")
plt.xlabel("Categories of type")
plt.ylabel("Transaction Counts")
plt.savefig("Graphs/Type Distribution.png")
plt.show()


#Department
plt.figure(figsize=(7 , 10))
sns.countplot(data=df , x = "Department Name" , edgecolor = "black")
plt.title("Distribution of Department")
plt.xlabel("Department")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.savefig("Graphs/Department Distribution.png")
plt.show()

#Market
plt.figure(figsize=(8 , 6 ))
colors = ["skyblue" , "lightgreen" , "yellow" , "orange" , "magenta"]
sns.countplot(data=df , x="Market" , edgecolor = "black" , palette=colors)
plt.title("Distribution of Market")
plt.xlabel("Market")
plt.ylabel("Count of Markets")
plt.xticks(rotation=0)
plt.savefig("Graphs/Market Distribution.png")
plt.show()


#order Country(Top 10 Countries)

country_counts = df["Order Country"].value_counts().head(10)
plt.figure(figsize=(8 , 6))
plt.pie(country_counts.values , labels=country_counts.index , autopct="%1.1f%%" )
plt.title("Top 10 Order Countries")
plt.savefig("Graphs/Order Country.png")
plt.show()


#Order status
order_status_counts = df["Order Status"].value_counts()
plt.figure(figsize=(8  , 6))
plt.pie(order_status_counts.values , labels=order_status_counts.index , autopct="%1.1f%%")
plt.title("Order Status")
plt.savefig("Graphs/Order Status.png")
plt.show()

#Top 10 product name
product_name_counts = df["Product Name"].value_counts().head(10)
plt.figure(figsize=(8 , 6))
plt.pie(product_name_counts.values , labels=product_name_counts.index , autopct= "%1.1f%%")
plt.title("Top 10 Product Names")
plt.savefig("Graphs/Product Names.png")
plt.show()


""""Bivariate Visualization"""

plt.figure(figsize=(8 , 6))
sns.countplot(data=df , x="Order Status" , hue="Late_delivery_risk" , palette="coolwarm" , edgecolor = "black")
plt.title("Late Delivery Risk Distribution by Order Status")
plt.xlabel("Order Status")
plt.ylabel("Count of Orders")
plt.xticks(rotation=45)  
plt.savefig("Graphs/Order Status Vs Risk.png")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,6))
sns.countplot(data=df , x="Shipping Mode" , hue="Late_delivery_risk" , palette="rocket" , edgecolor="black")
plt.title("Late Delivery Risk Distribution by Shipping Mode")
plt.xlabel("Shipping Mode")
plt.ylabel("Count of Orders")
plt.xticks(rotation=45)  
plt.savefig("Graphs/Shipping Mode Vs Risk.png")
plt.tight_layout()
plt.show()