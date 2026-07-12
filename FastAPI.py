import uvicorn
import joblib 
import pandas as pd 
from fastapi import FastAPI , HTTPException
from typing import List , Literal , Annotated
from pydantic import BaseModel , Field
from geopy.geocoders import Nominatim
from geopy.distance import geodesic


#Load the saved data
xgb_model = joblib.load("models/final_xgb_model.pkl")
artifacts = joblib.load("models/processed_data_pipeline.pkl")
X_train_cols = artifacts["X_train_encoded"].columns

#Geopy setup
geolocator = Nominatim(user_agent="smartfreight_v2_system" , timeout=10)
WAREHOUSE_COORDS = (28.6139, 77.2090) #Ware house at New Delhi

#FastAPI

app = FastAPI(
    title="SmartFreight",
    description="Shipment Delay Prediction"
)

#Request Schema
class ShipmentRequest(BaseModel):

    #Numerical Fields 
    sales_per_customer : Annotated[float , Field(ge = 10 , le= 2000 , description="Sales Per Customer (₹)" , examples=[50])]
    order_item_quantity : Annotated[float , Field(ge=1 , le = 5 , description="Order Quantity" , examples=[2])]
    product_price : Annotated[float , Field(ge = 10  , le=2000 , description="Product Price (₹)" , examples=[500])]
    destination_weather_risk : Annotated[float , Field(ge = 0.01 , le=0.99 ,description="Weather Risk (0.01 - 0.99)" , examples=[0.50])]

    #Types of transaction 
    type_of_transaction : Annotated[Literal["DEBIT", "TRANSFER", "CASH", "PAYMENT"] , Field(description="Type of transaction" , examples=["Cash"])]

    #Shipping mode
    shipping_mode : Annotated[Literal["Standard Class", "Second Class", "First Class", "Same Day"] , Field(description="Shipping Mode" , examples=["Standard Class"])]

    #Market
    market : Annotated[Literal["Africa", "Europe", "LATAM", "Pacific Asia", "USCA"] , Field(description="Market Region" , examples=["Market Region"])]

    #Destination city 
    destination_city : Annotated[str , Field(min_length=1 , max_length=50 , description="Destination City" , examples=["Kanpur"])]

    #Days Scheduled 
    days_scheduled : Annotated[int , Field (ge=0 , le=4 , description="Scheduled Shipping days" , examples=[3])]

    #Weather
    weather : Annotated[Literal["Clear", "Rainy", "Foggy", "Heavy_Snow"] , Field(description="Weather Condition" , examples=["Clear"])]

    #Category names - All 43 categories

    category_name : Annotated[
        Literal[
            "Accessories",
            "As Seen on TV",
            "Baby",
            "Baseball & Softball",
            "Basketball",
            "Books",
            "Cameras",
            "Camping & Hiking",
            "Cardio Equipment",
            "Children's Clothing",
            "Cleats",
            "Consumer Electronics",
            "Crafts",
            "Computers",
            "DVDs",
            "Electronics",
            "Fishing",
            "Fitness Accessories",
            "Garden",
            "Girls' Apparel",
            "Golf Bags & Carts",
            "Golf Balls",
            "Golf Gloves",
            "Golf Shoes",
            "Hunting & Shooting",
            "Indoor/Outdoor Games",
            "Kids' Golf Clubs",
            "Lacrosse",
            "Men's Footwear",
            "Music",
            "Musical Instruments",
            "Pet Supplies",
            "Product",
            "Record Players",
            "Robotics",
            "Soccer",
            "Sporting Goods",
            "Strength Training",
            "Striking Bags",
            "Tennis & Racquet",
            "Toys",
            "Trade In",
            "Video Games",
            "Water Sports",
            "Women's Apparel"
        ] , Field(description="Category Name" , examples=["Video Games"])
    ] 

    # Customer Segment
    customer_segment: Annotated[
        Literal["Consumer", "Corporate", "Home Office"],
        Field(description="Customer type" , examples=["Corporate"])
    ]

    #Customer Segment
    department : Annotated[
        Literal[
            "Apparel",
            "Book Fan",
            "Discs",
            "Fan Shop",
            "Fitness",
            "Footwear",
            "Golf",
            "Health and Beauty",
            "Language",
            "Outdoors",
            "Technology"
        ] , Field(description="Department" , examples=["Fitness"])
    ]

    

#Response Schema

class PredictResponse(BaseModel):
    delay_risk_percent : Annotated[float , Field(ge=0 , le=100 , description="Delay Probability (1-100)")]
    risk_level : Annotated[
        Literal["STABLE OPTIMAL ROUTE", "MODERATE CAUTION ZONE", "CRITICAL SLA RISK"] , Field (description="Latitude & Longitude")
    ]
    risk_delta: Annotated[str, Field(description="Risk vs baseline")] 
    calculated_distance_km: Annotated[float, Field(ge=0, description="Distance from warehouse (km)")] 
    city_coordinates: Annotated[dict, Field(description="Latitude & longitude")]
    model_confidence: Annotated[float, Field(ge=0, le=1, description="Model confidence (0-1)")]


@app.get("/")
def root():
    return{
        "message" : "Hello , User !!"
    }

@app.get("/about")
def about_page():
    return{
        "message" : "This model predicts the delayed shipment products."
    }

@app.get("/health")
def health():
    return{
        "message" : "Welcome to our model.",
        "Status" : 200 ,
        "Condition" : "running"
    }

#Predict

@app.post("/predict" , response_model=PredictResponse)
def predict_shipment_delayes(request : ShipmentRequest):
    try :
        location = geolocator.geocode(request.destination_city, timeout=10)
        if location is None:
            raise ValueError (f"City Not Found")
        
        city_coords = (location.latitude, location.longitude)
        distance_km = geodesic(WAREHOUSE_COORDS, city_coords).kilometers

        input_dict = {
            'Type': request.type_of_transaction,
            'Shipping Mode': request.shipping_mode,
            'Market': request.market,
            'Order City': request.destination_city,
            'Days for shipment (scheduled)': request.days_scheduled,
            'Weather': request.weather,
            'Category Name': request.category_name,
            'Customer Segment': request.customer_segment,
            'Departments': request.department,
            'Sales per customer': request.sales_per_customer,
            'Order Item Quantity': request.order_item_quantity,
            'Product Price': request.product_price,
            'Origin Weather Current': request.destination_weather_risk,
            'Dest Weather Risk Score': request.destination_weather_risk,
            'Distance_KM': distance_km
        }

        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=X_train_cols , fill_value=0.0)

        risk_prob = xgb_model.predict_proba(input_encoded)[0][1] * 100 

        if risk_prob >= 70:
            risk_level = "CRITICAL SLA RISK"

        elif risk_prob >= 35:
            risk_level = "MODERATE CAUTION ZONE"

        else :
            risk_level = "STABLE OPTIMAL ROUTE"

        risk_delta = risk_prob - 50.0
        delta_str = f"+{risk_delta:.1f}%" if risk_delta > 0 else f"{risk_delta:.1f}%"

        confidence = 1 - (abs(risk_prob -50 )/50)

        return PredictResponse(
            delay_risk_percent=round(risk_prob, 2),
            risk_level=risk_level,
            risk_delta=delta_str,
            calculated_distance_km=round(distance_km, 2),
            city_coordinates={"latitude": round(city_coords[0], 4), "longitude": round(city_coords[1], 4)},
            model_confidence=round(confidence, 3)
        )
    
    except Exception as e :
        raise HTTPException (status_code=500 , detail = str(e))