import os
import streamlit as st
import requests
import pydeck as pdk
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# ---------------------------------------------------------
# PAGE SETUP & BRANDING
# ---------------------------------------------------------
st.set_page_config(
    page_title="SmartFreight: Predictive Supply Chain",
    page_icon="📦",
    layout="wide"
)

FASTAPI_URL = "http://127.0.0.1:8000/predict"
WAREHOUSE_COORDS = (28.6139, 77.2090)  # Origin Hub: New Delhi HQ (Lat, Lon)

# Initialize Session State for cross-page route synchronization
if "route_data" not in st.session_state:
    st.session_state["route_data"] = {
        "city": "Mumbai",
        "coords": (19.0760, 72.8777),
        "distance_km": 1147.02,
        "risk_percent": 29.12,
        "base_risk": 27.04,
        "weather_penalty": 2.08,
        "risk_level": "STABLE OPTIMAL ROUTE",
        "weather_desc": "Dense Drizzle 🌧️",
        "market": "Pacific Asia",
        "country": "India"
    }

# ---------------------------------------------------------
# 1. HELPER: DATACO MARKET REGION DETECTOR
# ---------------------------------------------------------
def detect_market_region(country_code: str, country_name: str) -> str:
    c_code = (country_code or "").lower().strip()
    c_name = (country_name or "").lower().strip()

    if c_code in ["us", "ca", "pr"] or any(c in c_name for c in ["united states", "usa", "canada"]):
        return "USCA"

    pacific_asia_codes = {
        "in", "cn", "jp", "kr", "au", "nz", "sg", "my", "id", "th", "vn", "ph",
        "pk", "bd", "lk", "np", "tw", "hk", "ae", "sa", "qa", "kw", "om"
    }
    if c_code in pacific_asia_codes or any(c in c_name for c in ["india", "china", "japan", "australia", "korea", "singapore", "pakistan", "bangladesh", "thailand", "indonesia"]):
        return "Pacific Asia"

    europe_codes = {
        "gb", "uk", "de", "fr", "it", "es", "nl", "pl", "se", "no", "ch", "be",
        "at", "pt", "gr", "ie", "dk", "fi", "cz", "ro", "hu", "ua", "ru", "tr"
    }
    if c_code in europe_codes or any(c in c_name for c in ["united kingdom", "germany", "france", "italy", "spain", "netherlands", "russia", "turkey", "poland", "sweden", "switzerland"]):
        return "Europe"

    latam_codes = {
        "br", "mx", "ar", "co", "cl", "pe", "ve", "ec", "gt", "cu", "bo", "do", "pa", "cr", "uy", "py"
    }
    if c_code in latam_codes or any(c in c_name for c in ["brazil", "mexico", "argentina", "colombia", "chile", "peru", "venezuela", "cuba", "panama"]):
        return "LATAM"

    africa_codes = {
        "za", "eg", "ng", "ke", "gh", "ma", "dz", "et", "tz", "ug", "sn", "ci", "cm", "zw"
    }
    if c_code in africa_codes or any(c in c_name for c in ["south africa", "egypt", "nigeria", "kenya", "morocco", "ghana", "ethiopia"]):
        return "Africa"

    return "Pacific Asia"

# ---------------------------------------------------------
# 2. HELPER: ARRIVAL WEATHER FORECAST (OPEN-METEO)
# ---------------------------------------------------------
WMO_DESC = {
    0: ("Clear", 0.10, "Clear Sky ☀️"),
    1: ("Clear", 0.15, "Mainly Clear 🌤️"),
    2: ("Clear", 0.20, "Partly Cloudy ⛅"),
    3: ("Clear", 0.25, "Overcast ☁️"),
    45: ("Foggy", 0.50, "Heavy Fog 🌫️"),
    48: ("Foggy", 0.55, "Rime Fog 🌫️"),
    51: ("Rainy", 0.45, "Light Drizzle 🌦️"),
    53: ("Rainy", 0.55, "Moderate Drizzle 🌦️"),
    55: ("Rainy", 0.65, "Dense Drizzle 🌧️"),
    61: ("Rainy", 0.60, "Slight Rain 🌧️"),
    63: ("Rainy", 0.70, "Moderate Rain 🌧️"),
    65: ("Rainy", 0.85, "Heavy Downpour / Rainfall 🌧️"),
    71: ("Heavy_Snow", 0.70, "Snowfall 🌨️"),
    75: ("Heavy_Snow", 0.90, "Heavy Blizzard ❄️"),
    80: ("Rainy", 0.65, "Rain Showers 🌧️"),
    81: ("Rainy", 0.75, "Heavy Showers 🌧️"),
    82: ("Rainy", 0.90, "Torrential Storm ⛈️"),
    95: ("Rainy", 0.90, "Severe Thunderstorm ⚡"),
    99: ("Heavy_Snow", 0.95, "Severe Hailstorm ⛈️❄️")
}

def fetch_arrival_weather_forecast(lat: float, lon: float, days_scheduled: int):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=weathercode,temperature_2m_max,precipitation_sum,precipitation_probability_max,windspeed_10m_max&timezone=auto"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            daily = res.json().get("daily", {})
            times = daily.get("time", [])
            idx = min(days_scheduled, len(times) - 1)
            
            arrival_date = times[idx]
            wmo_code = daily.get("weathercode", [0])[idx]
            precip_mm = daily.get("precipitation_sum", [0.0])[idx]
            rain_prob = daily.get("precipitation_probability_max", [0])[idx]
            temp_max = daily.get("temperature_2m_max", [25.0])[idx]
            wind_max = daily.get("windspeed_10m_max", [10.0])[idx]

            category, risk_score, desc = WMO_DESC.get(wmo_code, ("Clear", 0.20, "Moderate Conditions 🌤️"))

            if precip_mm >= 25.0:
                risk_score = min(0.95, risk_score + 0.20)
                category = "Rainy"
            elif precip_mm >= 10.0:
                risk_score = min(0.90, risk_score + 0.10)
                category = "Rainy"

            return {
                "success": True,
                "arrival_date": arrival_date,
                "category": category,
                "risk_score": risk_score,
                "desc": desc,
                "precip_mm": precip_mm,
                "rain_prob": rain_prob,
                "temp": temp_max,
                "wind": wind_max
            }
    except Exception:
        pass

    fallback_date = (datetime.now() + timedelta(days=days_scheduled)).strftime("%Y-%m-%d")
    return {
        "success": False,
        "arrival_date": fallback_date,
        "category": "Clear",
        "risk_score": 0.20,
        "desc": "Clear (Telemetry Unavailable)",
        "precip_mm": 0.0,
        "rain_prob": 0,
        "temp": 26.0,
        "wind": 10.0
    }

# ---------------------------------------------------------
# SIDEBAR MULTI-PAGE NAVIGATION
# ---------------------------------------------------------
st.sidebar.title("📦 SmartFreight Portal")
st.sidebar.markdown("*Predictive Supply Chain Management*")
st.sidebar.write("---")

current_page = st.sidebar.radio(
    "Go To Navigation:",
    ["📄 Page 1: Risk Assessment Calculator", "🗺️ Page 2: 3D Route Telemetry Map"],
    index=0
)

st.sidebar.write("---")
st.sidebar.info(
    "💡 **Pro-Tip:** Predict a shipment on **Page 1**, then switch to **Page 2** to inspect the live 3D parabolic flight arc & routing telemetry!"
)


# =========================================================
# PAGE 1: PREDICTIVE RISK ASSESSMENT CALCULATOR
# =========================================================
if current_page == "📄 Page 1: Risk Assessment Calculator":
    st.title("📦 SmartFreight: Predictive Supply Chain Dashboard")
    st.markdown("A real-time Machine Learning pipeline predicting shipment delays to prevent logistics SLA breaches.")
    st.write("---")

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.subheader("Order & Cost 📦")
        sales_per_customer = st.number_input("Sales Per Customer (₹)", min_value=10, max_value=2000, value=50)
        order_item_quantity = st.number_input("Order Quantity", min_value=1, max_value=5, value=1)
        product_price = st.number_input("Product Price (₹)", min_value=10, max_value=2000, value=10)
        customer_segment = st.selectbox("Customer Segment", ["Consumer", "Corporate", "Home Office"])

    calculated_distance_km = 500.0
    destination_coords = (19.0760, 72.8777)
    arrival_weather_info = None
    detected_market = "Pacific Asia"
    detected_country = "India"

    with col2:
        st.subheader("Route & Arrival Weather 🛰️")
        destination_city = st.text_input("Enter Destination City (e.g. Mumbai, New York, London)", value="Mumbai")

        if destination_city.strip():
            try:
                geolocator = Nominatim(user_agent="smartfreight_arrival_tracker_v6", timeout=10)
                location = geolocator.geocode(destination_city.strip(), timeout=10, addressdetails=True, language="en")

                if location is not None:
                    destination_coords = (location.latitude, location.longitude)
                    calculated_distance_km = round(geodesic(WAREHOUSE_COORDS, destination_coords).kilometers, 2)

                    address_dict = location.raw.get("address", {})
                    detected_country = address_dict.get("country", "Unknown")
                    country_code = address_dict.get("country_code", "")
                    detected_market = detect_market_region(country_code, detected_country)

                    st.success(f"📍 **Destination:** {location.address}")
                    st.markdown(
                        f"""
                        <div style="background-color:#1e293b; padding:10px 14px; border-radius:8px; margin-bottom:12px;">
                            <span style="color:#94a3b8; font-size:13px;">Transit Distance from Delhi Hub:</span>
                            <h4 style="margin:2px 0 0 0; color:#38bdf8;">{calculated_distance_km:,.2f} KM</h4>
                            <span style="color:#a7f3d0; font-size:12px;">🌍 Auto-detected Region: <b>{detected_market}</b> ({detected_country})</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    days_scheduled_preview = 2
                    arrival_weather_info = fetch_arrival_weather_forecast(
                        location.latitude, location.longitude, days_scheduled_preview
                    )

                    rain_badge = f"🌧️ Rain: {arrival_weather_info['precip_mm']}mm ({arrival_weather_info['rain_prob']}% chance)" if arrival_weather_info['precip_mm'] > 0 else "☀️ No Rain Expected"
                    
                    st.markdown(
                        f"""
                        <div style="background-color:#0f172a; border-left: 4px solid #f59e0b; padding:12px; border-radius:6px; margin-top:5px;">
                            <b style="color:#fbbf24; font-size:14px;">📅 Arrival Day Weather Forecast ({arrival_weather_info['arrival_date']}):</b><br>
                            <span style="font-size:15px; color:#f8fafc;">• <b>Condition:</b> {arrival_weather_info['desc']}</span><br>
                            <span style="font-size:14px; color:#cbd5e1;">• <b>Precipitation:</b> {rain_badge}</span><br>
                            <span style="font-size:14px; color:#94a3b8;">• <b>Temperature:</b> {arrival_weather_info['temp']}°C | <b>Wind:</b> {arrival_weather_info['wind']} km/h</span><br>
                            <span style="font-size:14px; color:#38bdf8;">• <b>Arrival Weather Risk Factor:</b> {arrival_weather_info['risk_score']:.2f}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:
                    st.warning("City not recognized. Using default baseline distance (500 KM).")
            except Exception:
                st.warning("⚠️ Geocoding service busy. Using standard baseline routing.")

        category_name = st.selectbox("Product Category", [
            "Accessories", "As Seen on TV", "Baby", "Baseball & Softball", "Basketball",
            "Books", "Cameras", "Camping & Hiking", "Cardio Equipment", "Children's Clothing",
            "Cleats", "Consumer Electronics", "Crafts", "Computers", "DVDs",
            "Electronics", "Fishing", "Fitness Accessories", "Garden", "Girls' Apparel",
            "Golf Bags & Carts", "Golf Balls", "Golf Gloves", "Golf Shoes", "Hunting & Shooting",
            "Indoor/Outdoor Games", "Kids' Golf Clubs", "Lacrosse", "Men's Footwear", "Music",
            "Musical Instruments", "Pet Supplies", "Product", "Record Players", "Robotics",
            "Soccer", "Sporting Goods", "Strength Training", "Striking Bags", "Tennis & Racquet",
            "Toys", "Trade In", "Video Games", "Water Sports", "Women's Apparel"
        ])

    with col3:
        st.subheader("Shipping Logistics 🚚")
        transaction_type = st.selectbox("Type Of Transaction", ["DEBIT", "TRANSFER", "CASH", "PAYMENT"])
        shipping_mode = st.selectbox("Shipping Mode", ["Standard Class", "Second Class", "First Class", "Same Day"])

        market_options = ["Pacific Asia", "USCA", "Europe", "LATAM", "Africa"]
        default_market_idx = market_options.index(detected_market) if detected_market in market_options else 0
        market = st.selectbox("Market Region", market_options, index=default_market_idx)
        st.caption(f"⚡ Auto-assigned to **{market}** based on `{destination_city.title()}` ({detected_country})")

        departments = st.selectbox("Department", [
            "Apparel", "Book Fan", "Discs", "Fan Shop", "Fitness",
            "Footwear", "Golf", "Health and Beauty", "Language", "Outdoors", "Technology"
        ])
        days_scheduled = st.slider("Scheduled Shipping Days (Transit Time)", min_value=1, max_value=5, value=2)

    # Prediction Action
    st.write("")
    if st.button("🚀 Predict Shipment Delay Risk", use_container_width=True):
        if days_scheduled <= 0:
            st.error(f"🚫 **Invalid Schedule: Zero-Day Delivery is not possible!** Intercity transit across {calculated_distance_km:,.1f} KM requires at least 1 scheduled shipping day.")
            st.stop()
        elif not destination_city.strip() or arrival_weather_info is None:
            st.error("Please enter a valid destination city to fetch route and weather telemetry.")
            st.stop()
        else:
            payload = {
                "sales_per_customer": float(sales_per_customer),
                "order_item_quantity": float(order_item_quantity),
                "product_price": float(product_price),
                "destination_weather_risk": float(arrival_weather_info["risk_score"]),
                "type_of_transaction": transaction_type,
                "shipping_mode": shipping_mode,
                "market": market,
                "destination_city": destination_city.strip(),
                "days_scheduled": int(days_scheduled),
                "weather": arrival_weather_info["category"],
                "category_name": category_name,
                "customer_segment": customer_segment,
                "department": departments,
                "distance_km": float(calculated_distance_km)
            }

            data = None
            try:
                with st.spinner(f"Scoring delay risk for {destination_city.title()} ({market} route)..."):
                    response = requests.post(FASTAPI_URL, json=payload, timeout=3)
                    if response.status_code == 200:
                        data = response.json()
            except Exception:
                pass

            # Smart Fallback for Streamlit Cloud
            if data is None:
                try:
                    import joblib
                    import pandas as pd
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    xgb_model = joblib.load(os.path.join(base_dir, "models", "final_xgb_model.pkl"))
                    artifacts = joblib.load(os.path.join(base_dir, "models", "processed_data_pipeline.pkl"))
                    X_train_cols = artifacts["X_train_encoded"].columns

                    base_dict = {
                        'Type': transaction_type, 'Shipping Mode': shipping_mode, 'Market': market,
                        'Order City': destination_city.strip(), 'Days for shipment (scheduled)': days_scheduled,
                        'Category Name': category_name, 'Customer Segment': customer_segment,
                        'Department Name': departments, 'Departments': departments,
                        'Sales per customer': float(sales_per_customer), 'Order Item Quantity': float(order_item_quantity),
                        'Product Price': float(product_price), 'Distance_KM': float(calculated_distance_km)
                    }

                    def predict_risk(weather_cond, weather_score):
                        d = base_dict.copy()
                        d['Weather'] = weather_cond
                        d['Dest_Weather_Risk_Score'] = weather_score
                        d['Dest Weather Risk Score'] = weather_score
                        d['Origin Weather Current'] = weather_score
                        df = pd.DataFrame([d])
                        enc = pd.get_dummies(df).reindex(columns=X_train_cols, fill_value=0.0)
                        return round(float(xgb_model.predict_proba(enc)[0][1] * 100.0), 2)

                    base_risk = predict_risk("Clear", 0.10)
                    actual_risk = predict_risk(arrival_weather_info["category"], arrival_weather_info["risk_score"])
                    weather_penalty = max(0.0, round(actual_risk - base_risk, 2))
                    risk_level = "CRITICAL SLA RISK" if actual_risk >= 70.0 else ("MODERATE CAUTION ZONE" if actual_risk >= 35.0 else "STABLE OPTIMAL ROUTE")
                    delta_val = actual_risk - 50.0
                    delta_str = f"+{delta_val:.1f}%" if delta_val > 0 else f"{delta_val:.1f}%"

                    explanation = f"⚠️ Delay probability increased by +{weather_penalty:.1f}% due to adverse weather ({arrival_weather_info['category']}) forecasted on arrival." if weather_penalty > 0 else "✅ Optimal weather conditions forecasted upon arrival; 0% delay penalty."

                    data = {
                        "delay_risk_percent": actual_risk, "base_operational_risk": base_risk,
                        "weather_penalty_percent": weather_penalty, "risk_level": risk_level,
                        "risk_delta": delta_str, "model_confidence": round(abs(actual_risk - 50.0) / 50.0, 2),
                        "weather_explanation": explanation
                    }
                except Exception as e:
                    st.error(f"Inference error: {str(e)}")

            if data:
                # Update Session State for Page 2
                st.session_state["route_data"] = {
                    "city": destination_city.title(),
                    "coords": destination_coords,
                    "distance_km": calculated_distance_km,
                    "risk_percent": data["delay_risk_percent"],
                    "base_risk": data["base_operational_risk"],
                    "weather_penalty": data["weather_penalty_percent"],
                    "risk_level": data["risk_level"],
                    "weather_desc": arrival_weather_info["desc"],
                    "market": market,
                    "country": detected_country
                }

                total_risk = data["delay_risk_percent"]
                base_risk = data["base_operational_risk"]
                weather_penalty = data["weather_penalty_percent"]
                risk_level = data["risk_level"]
                risk_delta_str = data["risk_delta"]
                confidence = data["model_confidence"]
                explanation = data["weather_explanation"]

                st.markdown("### 📊 Live Risk Assessment & Root Cause Breakdown")
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Delay Probability", f"{total_risk:.2f}%", delta=f"{risk_delta_str} vs Baseline", delta_color="inverse")
                m2.metric("Base Operational Transit Risk", f"{base_risk:.2f}%")
                m3.metric("Weather Delay Penalty", f"+{weather_penalty:.2f}%", delta=f"{arrival_weather_info['category']} Alert" if weather_penalty > 0 else "Clear Skies", delta_color="inverse")

                if weather_penalty > 0:
                    st.warning(f"{explanation}\n\n**Rainfall Expected:** `{arrival_weather_info['precip_mm']} mm` ({arrival_weather_info['rain_prob']}% chance) on **{arrival_weather_info['arrival_date']}**.")
                else:
                    st.success(explanation)

                st.markdown("### 🗺️ Live Operational Trajectory")
                badge_color = "#d9534f" if total_risk >= 70.0 else ("#f0ad4e" if total_risk >= 35.0 else "#5cb85c")
                st.markdown(f'<span style="background-color:{badge_color}; color:white; padding:6px 14px; border-radius:4px; font-weight:bold; font-size:15px;">{risk_level}</span>', unsafe_allow_html=True)
                st.progress(max(0, min(100, int(total_risk))))
                st.caption(f"System Matrix Factor: {total_risk:.2f}% | Model Confidence: {confidence * 100:.0f}%")

                st.write("")
                st.info("👉 **Switch to Page 2 in the left sidebar to view the full-screen 3D Geospatial Transit Map!**")


# =========================================================
# PAGE 2: DEDICATED 3D GEOSPATIAL TRANSIT MAP
# =========================================================
elif current_page == "🗺️ Page 2: 3D Route Telemetry Map":
    st.title("🗺️ 3D Geospatial Route & Telemetry Center")
    st.markdown("Global 3D flight and freight corridor visualization powered by WebGL & Pydeck.")
    st.write("---")

    active_route = st.session_state["route_data"]
    dest_coords = active_route["coords"]
    total_risk = active_route["risk_percent"]
    risk_level = active_route["risk_level"]
    distance_km = active_route["distance_km"]

    # Top KPI Metrics on Page 2
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Origin Hub", "New Delhi HQ (DEL)", "Primary Dispatch Hub")
    kpi2.metric(f"Destination: {active_route['city']}", f"{distance_km:,.1f} KM", f"{active_route['market']} Region")
    kpi3.metric("Live Delay Probability", f"{total_risk:.2f}%", active_route["risk_level"], delta_color="inverse")
    kpi4.metric("Weather Impact", f"+{active_route['weather_penalty']:.2f}%", active_route["weather_desc"])

    st.write("")

    # Map Arc Color Definition
    if total_risk >= 70.0:
        arc_rgb = [217, 83, 79, 240]       # Red
    elif total_risk >= 35.0:
        arc_rgb = [240, 173, 78, 240]      # Orange
    else:
        arc_rgb = [92, 184, 92, 240]        # Green

    origin_coords_lonlat = [WAREHOUSE_COORDS[1], WAREHOUSE_COORDS[0]]
    dest_coords_lonlat = [dest_coords[1], dest_coords[0]]

    # 3D Arc Dataset
    arc_data = [{
        "from": origin_coords_lonlat,
        "to": dest_coords_lonlat,
        "name": f"Freight Corridor: New Delhi ➔ {active_route['city']}",
        "info": f"Transit Distance: {distance_km:,.1f} KM | Risk: {total_risk:.1f}% ({risk_level})"
    }]

    # Markers Dataset
    scatter_data = [
        {
            "position": origin_coords_lonlat,
            "name": "🏭 Central Warehouse: New Delhi HQ",
            "info": "Dispatch Coordinates: (28.6139° N, 77.2090° E)",
            "color": [56, 189, 248, 240],   # Bright Cyan
            "radius": 40000
        },
        {
            "position": dest_coords_lonlat,
            "name": f"📍 Destination Hub: {active_route['city']}, {active_route['country']}",
            "info": f"Delay Probability: {total_risk:.1f}% | Weather: {active_route['weather_desc']}",
            "color": arc_rgb,               # Dynamic Risk Color
            "radius": 50000
        }
    ]

    arc_layer = pdk.Layer(
        "ArcLayer",
        data=arc_data,
        get_source_position="from",
        get_target_position="to",
        get_source_color=[56, 189, 248, 220],
        get_target_color=arc_rgb,
        get_width=6,
        auto_highlight=True,
        pickable=True
    )

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=scatter_data,
        get_position="position",
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
        auto_highlight=True
    )

    # Dynamic camera positioning
    mid_lat = (WAREHOUSE_COORDS[0] + dest_coords[0]) / 2.0
    mid_lon = (WAREHOUSE_COORDS[1] + dest_coords[1]) / 2.0

    if distance_km < 600:
        zoom_lvl = 5.5
    elif distance_km < 1600:
        zoom_lvl = 4.2
    elif distance_km < 4000:
        zoom_lvl = 3.0
    elif distance_km < 8000:
        zoom_lvl = 2.0
    else:
        zoom_lvl = 1.3

    view_state = pdk.ViewState(
        latitude=mid_lat,
        longitude=mid_lon,
        zoom=zoom_lvl,
        pitch=48,  # 3D tilted camera angle
        bearing=0
    )

    deck_chart = pdk.Deck(
        layers=[arc_layer, scatter_layer],
        initial_view_state=view_state,
        map_style="dark",
        tooltip={"text": "{name}\n{info}"}
    )

    # Full-width 3D Map Rendering on Page 2
    st.pydeck_chart(deck_chart)