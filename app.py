import pandas as pd 
import streamlit as st
import joblib
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

artifacts = joblib.load("models/processed_data_pipeline.pkl")
model = joblib.load("models/final_xgb_model.pkl")

st.title("📦 SmartFreight: Predictive Supply Chain Dashboard")
st.markdown("*A real-time Machine Learning pipeline predicting shipment delays to prevent logistics SLA breaches.*")
st.markdown("---")

#Creating columns 
col1 , col2 , col3= st.columns(3 , gap="large")

#Numerical column(Col1)
with col1 : 
    Sales_per_customer = st.number_input("Sales Per Customer" , min_value=10 , max_value=2000 , value=50)
    Order_item_quantity = st.number_input("Order Quantity" , min_value=1 , max_value=5 , value=1)
    Product_Price = st.number_input("Product Price" , min_value=10 , max_value = 2000 , value=10)
    Dest_weather_risk_score = st.number_input("Destinaton Weather" , min_value= 0.1 , max_value=0.9 , value=0.1)
    st.caption("ℹ️ Note: 0.1 to 0.3 = Clear, 0.4 to 0.6 = Rainy/Foggy, 0.7 to 0.9 = Severe Storm (High Delay Risk).")

with col2 :
    type = st.selectbox("Type Of Transaction" , ["DEBIT", "TRANSFER", "CASH", "PAYMENT"])

    st.subheader("Route Configuration")

    #Manual city input
    user_city = st.text_input("Enter the destination city")

    #Default value 
    distance_Km = 5000.0

    #Calculate distance if user types a city 
    if user_city:
        locator = Nominatim(user_agent="my_simple_app")
        location = locator.geocode(user_city)

        if location is not None:
            city_coords = (location.latitude , location.longitude)
            warehouse_coordinates = (28.6139, 77.2090)


            #Calculate the real life distance 
            distance_Km = geodesic(warehouse_coordinates , city_coords).kilometers

            st.success(f"City Found: {location.address}")
            st.metric(label="Calculated Shipping Distance" , value=f"{distance_Km:.2f} KM")

        else :
            st.error("City Not Found , try with the another city that is nearest to destination.")

    category_name = st.selectbox("Category" , ["Accessories", "As Seen on TV!", "Baby ", "Baseball & Softball", "Basketball", "Books", "Cameras", "Camping & Hiking", "Cardio Equipment", "Children's Clothing", "Cleats", "Consumer Electronics", "Crafts", "Computers", "DVDs", "Electronics", "Fishing", "Fitness Accessories", "Garden", "Girls' Apparel", "Golf Bags & Carts", "Golf Balls", "Golf Gloves", "Golf Shoes", "Hunting & Shooting", "Indoor/Outdoor Games", "Kids' Golf Clubs", "Lacrosse", "Men's Footwear", "Music", "Musical Instruments", "Pet Supplies", "Product", "Record Players", "Robotics", "Soccer", "Sporting Goods", "Strength Training", "Striking Bags", "Tennis & Racquet", "Toys", "Trade In", "Video Games", "Water Sports", "Women's Apparel"])

    customer_segment = st.selectbox("Customer Segment" , ["Consumer" , "Corporate" , "Home Office"])

    departments = st.selectbox("Departments" , ["Apparel", "Book Fan", "Discs", "Fan Shop", "Fitness", "Footwear", "Golf", "Health and Beauty ", "Language", "Outdoors", "Technology"])

with col3:
    shipping_mode = st.selectbox("Shipping Mode" , ["Standard Class", "Second Class", "First Class", "Same Day"])
    market = st.selectbox("Market" , ["Africa", "Europe", "LATAM", "Pacific Asia", "USCA"])
    days_scheduled = st.slider("Scheduled Shipping Days" , min_value=0  , max_value=4 , value=1)
    weather = st.selectbox("Weather" , ["Clear", "Rainy", "Foggy", "Heavy_Snow"])


if st.button("🚀 Predict Shipment Delay Risk", use_container_width=True):
    input_df = pd.DataFrame([{
        'Type': type,
        'Shipping Mode': shipping_mode,
        'Order City': user_city,
        'Category Name': category_name,
        'Customer Segment': customer_segment,
        'Department Name': departments,
        'Market': market,
        'Days for shipment (scheduled)': days_scheduled,
        'Sales per customer': Sales_per_customer,
        'Order Item Quantity': Order_item_quantity,
        'Product Price': Product_Price,
        'Origin_Weather_Current': weather,
        'Dest_Weather_Risk_Score': Dest_weather_risk_score,
        'Distance_KM': distance_Km
    }])

    
    # 1. Direct one-hot encoding bina kisi transform complexity ke
    input_encoded = pd.get_dummies(input_df)
    
    # 2. X_train ki columns se pure matrix dimensions ko direct force-align karo
    X_train_cols = artifacts["X_train_encoded"].columns
    input_encoded = input_encoded.reindex(columns=X_train_cols, fill_value=0.0)
    
    # 3. Final risk calculation via XGBoost matrix state
    risk_percent = model.predict_proba(input_encoded)[0][1] * 100
    
    # --- DYNAMIC RESPONSE LAYER (INSTAGRAM TYPE THEME BLOCKS) ---
    st.markdown("### 📊 Live Risk Assessment")
    st.metric(label="Calculated Delay Probability", value=f"{risk_percent:.2f}%", delta=f"{risk_percent - 50:.1f}% Risk Delta", delta_color="inverse")
    # 1. Float mapping structure ko integer progress percentage me convert kiya
    progress_val = int(risk_percent)
    
    # 2. Progress value bounds controller (Streamlit ka strict check: Value must be 0 to 100)
    if progress_val > 100: progress_val = 100
    if progress_val < 0: progress_val = 0

    st.markdown("### 🗺️ Live Operational Trajectory")
    
    # 3. Dynamic badge and progress bar coloration logic based on risk intervals
    if risk_percent >= 70.0:
        st.markdown("<span style='background-color:#ef4444; color:white; padding:4px 8px; border-radius:4px; font-weight:bold;'>CRITICAL SLA RISK</span>", unsafe_allow_html=True)
        st.progress(progress_val) # Standard Red component container trigger automatically
    elif 35.0 <= risk_percent < 70.0:
        st.markdown("<span style='background-color:#f59e0b; color:black; padding:4px 8px; border-radius:4px; font-weight:bold;'>MODERATE CAUTION ZONE</span>", unsafe_allow_html=True)
        st.progress(progress_val)
    else:
        st.markdown("<span style='background-color:#10b981; color:white; padding:4px 8px; border-radius:4px; font-weight:bold;'>STABLE OPTIMAL ROUTE</span>", unsafe_allow_html=True)
        st.progress(progress_val)

    # 4. Secondary micro-text breakdown
    st.caption(f"Current System Matrix Deviation Factor: **{risk_percent:.2f}%**")