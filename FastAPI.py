import os
import uvicorn
import joblib
import pandas as pd
from typing import Literal, Annotated, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# 1. SETUP & MODEL LOADING (Absolute Paths)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_xgb_model.pkl")
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "processed_data_pipeline.pkl")

try:
    xgb_model = joblib.load(MODEL_PATH)
    artifacts = joblib.load(PIPELINE_PATH)
    X_train_cols = artifacts["X_train_encoded"].columns
except Exception as e:
    raise RuntimeError(f"Error loading models from {BASE_DIR}/models: {str(e)}")

app = FastAPI(
    title="SmartFreight API",
    description="REST API for Predictive Supply Chain & Weather-Induced Delay Scoring",
    version="2.2.0"
)

# Enable CORS for external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 2. REQUEST & RESPONSE SCHEMAS
# ---------------------------------------------------------
CATEGORY_LITERAL = Literal[
    "Accessories", "As Seen on TV", "Baby", "Baseball & Softball", "Basketball",
    "Books", "Cameras", "Camping & Hiking", "Cardio Equipment", "Children's Clothing",
    "Cleats", "Consumer Electronics", "Crafts", "Computers", "DVDs",
    "Electronics", "Fishing", "Fitness Accessories", "Garden", "Girls' Apparel",
    "Golf Bags & Carts", "Golf Balls", "Golf Gloves", "Golf Shoes", "Hunting & Shooting",
    "Indoor/Outdoor Games", "Kids' Golf Clubs", "Lacrosse", "Men's Footwear", "Music",
    "Musical Instruments", "Pet Supplies", "Product", "Record Players", "Robotics",
    "Soccer", "Sporting Goods", "Strength Training", "Striking Bags", "Tennis & Racquet",
    "Toys", "Trade In", "Video Games", "Water Sports", "Women's Apparel"
]

class ShipmentRequest(BaseModel):
    sales_per_customer: Annotated[float, Field(ge=10, le=2000, description="Sales Per Customer (₹)", examples=[50.0])]
    order_item_quantity: Annotated[float, Field(ge=1, le=5, description="Order Quantity", examples=[1.0])]
    product_price: Annotated[float, Field(ge=10, le=2000, description="Product Price (₹)", examples=[10.0])]
    destination_weather_risk: Annotated[float, Field(ge=0.01, le=0.99, description="Weather Risk Score (0.01 - 0.99)", examples=[0.10])]
    type_of_transaction: Annotated[Literal["DEBIT", "TRANSFER", "CASH", "PAYMENT"], Field(description="Transaction type", examples=["DEBIT"])]
    shipping_mode: Annotated[Literal["Standard Class", "Second Class", "First Class", "Same Day"], Field(description="Shipping mode", examples=["Standard Class"])]
    market: Annotated[Literal["Africa", "Europe", "LATAM", "Pacific Asia", "USCA"], Field(description="Market Region", examples=["Africa"])]
    destination_city: Annotated[str, Field(min_length=1, max_length=100, description="Destination City", examples=["Chennai"])]
    # ZERO-DAY DELIVERY VALIDATION: Minimum scheduled days must be at least 1
    days_scheduled: Annotated[int, Field(ge=1, le=6, description="Scheduled Shipping Days (must be >= 1)", examples=[2])]
    weather: Annotated[Literal["Clear", "Rainy", "Foggy", "Heavy_Snow"], Field(description="Arrival Weather Condition", examples=["Clear"])]
    category_name: Annotated[CATEGORY_LITERAL, Field(description="Product Category", examples=["Accessories"])]
    customer_segment: Annotated[Literal["Consumer", "Corporate", "Home Office"], Field(description="Customer Segment", examples=["Consumer"])]
    department: Annotated[Literal[
        "Apparel", "Book Fan", "Discs", "Fan Shop", "Fitness",
        "Footwear", "Golf", "Health and Beauty", "Language", "Outdoors", "Technology"
    ], Field(description="Department Name", examples=["Apparel"])]
    distance_km: Annotated[float, Field(ge=0, description="Distance from warehouse in KM", examples=[350.0])]

class PredictResponse(BaseModel):
    delay_risk_percent: float = Field(..., description="Overall Delay Probability (%)")
    base_operational_risk: float = Field(..., description="Base delay risk under clear weather (%)")
    weather_penalty_percent: float = Field(..., description="Additional delay risk directly attributed to weather (%)")
    risk_level: Literal["STABLE OPTIMAL ROUTE", "MODERATE CAUTION ZONE", "CRITICAL SLA RISK"]
    risk_delta: str = Field(..., description="Deviation against 50% baseline")
    calculated_distance_km: float = Field(..., description="Transit distance in KM")
    model_confidence: float = Field(..., description="Model certainty factor (0.0 - 1.0)")
    weather_explanation: str = Field(..., description="Human-readable root cause explanation")

# ---------------------------------------------------------
# 3. HELPER INFERENCE FUNCTION
# ---------------------------------------------------------
def run_model_inference(payload_dict, weather_condition, weather_score):
    d = payload_dict.copy()
    d['Weather'] = weather_condition
    d['Dest_Weather_Risk_Score'] = weather_score
    d['Dest Weather Risk Score'] = weather_score
    d['Origin Weather Current'] = weather_score
    
    df = pd.DataFrame([d])
    encoded = pd.get_dummies(df).reindex(columns=X_train_cols, fill_value=0.0)
    prob = float(xgb_model.predict_proba(encoded)[0][1] * 100.0)
    return round(prob, 2)

# ---------------------------------------------------------
# 4. API ENDPOINTS
# ---------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {"message": "SmartFreight Prediction API is running", "status": "online"}

@app.get("/health", tags=["Health"])
def health():
    return {"status": 200, "condition": "operational", "model_loaded": True}

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_shipment(request: ShipmentRequest):
    # ZERO-DAY VALIDATION GATE: Halt impossible transit schedules
    if request.days_scheduled <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Zero-day delivery is physically impossible for a transit distance of {request.distance_km:.1f} KM. Scheduled shipping days must be at least 1."
        )

    try:
        base_dict = {
            'Type': request.type_of_transaction,
            'Shipping Mode': request.shipping_mode,
            'Market': request.market,
            'Order City': request.destination_city,
            'Days for shipment (scheduled)': request.days_scheduled,
            'Category Name': request.category_name,
            'Customer Segment': request.customer_segment,
            'Department Name': request.department,
            'Departments': request.department,
            'Sales per customer': request.sales_per_customer,
            'Order Item Quantity': request.order_item_quantity,
            'Product Price': request.product_price,
            'Distance_KM': request.distance_km
        }

        # 1. Base operational prediction (assuming clear weather baseline)
        base_risk = run_model_inference(base_dict, "Clear", 0.10)

        # 2. Actual prediction with forecasted arrival weather
        actual_risk = run_model_inference(base_dict, request.weather, request.destination_weather_risk)

        # 3. Calculate weather penalty
        weather_penalty = max(0.0, round(actual_risk - base_risk, 2))

        # Risk level classification
        if actual_risk >= 70.0:
            risk_level = "CRITICAL SLA RISK"
        elif actual_risk >= 35.0:
            risk_level = "MODERATE CAUTION ZONE"
        else:
            risk_level = "STABLE OPTIMAL ROUTE"

        # Baseline deviation delta
        delta = actual_risk - 50.0
        delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        confidence = round(abs(actual_risk - 50.0) / 50.0, 2)

        # Weather impact explanation
        if weather_penalty > 10.0:
            explanation = f"⚠️ Delay probability increased by +{weather_penalty:.1f}% due to adverse weather ({request.weather}) forecasted on arrival."
        elif weather_penalty > 0.0:
            explanation = f"ℹ️ Minor delay impact of +{weather_penalty:.1f}% attributed to local weather conditions on arrival."
        else:
            explanation = "✅ Optimal weather conditions forecasted upon arrival; 0% delay penalty."

        return PredictResponse(
            delay_risk_percent=actual_risk,
            base_operational_risk=base_risk,
            weather_penalty_percent=weather_penalty,
            risk_level=risk_level,
            risk_delta=delta_str,
            calculated_distance_km=request.distance_km,
            model_confidence=confidence,
            weather_explanation=explanation
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("FastAPI:app", host="127.0.0.1", port=8000, reload=True)