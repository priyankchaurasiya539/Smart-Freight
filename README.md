# 📦 SmartFreight: Predictive Supply Chain Dashboard

This project is a web-based dashboard built with **Streamlit** and **Machine Learning (XGBoost)**. It helps logistics managers predict if a shipment will be delayed or delivered on time before it even leaves the warehouse. 

By predicting delays early, companies can avoid breaking their **SLA (Service Level Agreement)** promises and save money on penalties.

---

## 📈 Model Performance (How accurate is it?)

The brain of this dashboard is a trained AI model that looks at historical shipping records, weather, and order details to calculate risks.

* **Prediction Accuracy:** **89.40%** (It guesses correctly roughly 9 out of 10 times)
* **ROC-AUC Score:** **0.91** (This means the model is highly reliable at separating high-risk shipments from safe ones)

---

## 🚀 Key Features

* **Instant Risk Calculator:** Enter details like city, price, or weather, and get a risk percentage instantly.
* **Compact 3-Column Layout:** All inputs fit perfectly on a single screen without annoying scrolling.
* **Smart Alert Colors:** The output automatically changes color based on the risk level (Green for safe, Orange for warning, Red for high danger).
* **Risk Delta Counter:** Shows you exactly how much safer or riskier your current shipment is compared to a normal baseline.

---

## 📁 Project Folder Structure

```text
SmartFreight/
├── data/
│   └── DataCoSupplyChainDataset.csv  # The main data used to train the AI
├── Graphs/
│   ├── Department Distribution.png   # Charts generated from the EDA pipeline
│   ├── Market Distribution.png
│   ├── Order Country.png
│   ├── Order Status Vs Risk.png
│   ├── Order Status.png
│   ├── Product Names.png
│   ├── Shipping Mode Vs Risk.png
│   └── Type Distribution.png
├── models/
│   ├── final_xgb_model.pkl          # The updated saved trained XGBoost model file
│   └── processed_data_pipeline.pkl   # Saves column structure settings
├── .gitattributes
├── .gitignore
├── app.py                            # The code that runs the Streamlit website UI
├── EDA.py                            # Python code for Exploratory Data Analysis
├── feature_data.py                   # Data preprocessing and feature engineering steps
├── model_training.py                 # The Python code used to train the AI
└── requirements.txt                  # List of required packages to run the app
