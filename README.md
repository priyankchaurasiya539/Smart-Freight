# 📦 SmartFreight: Predictive Supply Chain & Delay Risk Engine

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20Classifier-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A real-time predictive logistics microservice that flags shipment delay risks and SLA breach vulnerabilities before packages even leave the warehouse.

---

## 🚀 Overview

In modern global logistics, transit delays breach strict **Service Level Agreements (SLAs)**, leading to severe contract penalties, customer dissatisfaction, and operational gridlock. 

**SmartFreight** transforms reactive package tracking into **proactive delay prevention**. By analyzing operational order attributes, geographic routing, and real-time arrival-day satellite weather forecasts, SmartFreight instantly predicts delay probabilities and breaks down root causes for logistics operators before dispatch.

---

## ✨ Key Features

- **Decoupled Client-Server Architecture**: Dedicated **FastAPI REST backend** (`POST /predict`) with strict Pydantic schemas, completely decoupled from the **Streamlit** dashboard.
- **🛰️ Dynamic Arrival-Day Weather Telemetry**: Queries Open-Meteo satellite APIs for destination coordinates on the *exact expected arrival date* ($\text{Today} + \text{Transit Days}$) rather than relying on stale current-day weather.
- **🔍 Root Cause Risk Attribution**: Separates **Base Operational Transit Risk** from **Weather Delay Penalty (+X%)**, directly exposing how rainfall, storms, or blizzards contribute to SLA breach risks.
- **🌍 Automated Geospatial & Market Mapping**: 
  - Geocodes destination cities via Geopy and calculates geodesic transit distances from origin hubs (Delhi HQ).
  - Automatically identifies destination country and maps it to the corresponding global market region (**Pacific Asia, USCA, Europe, LATAM, Africa**), eliminating operator misconfiguration.
- **🛡️ Physical Feasibility Enforcement**: Rejects physically impossible inputs (such as zero-day intercity transit schedules) at both the client UI and API schema validation levels.
- **🎨 Responsive Operational Dashboard**: Compact 3-column operational layout with live risk delta counters and color-coded trajectory badges (**STABLE OPTIMAL ROUTE**, **MODERATE CAUTION ZONE**, **CRITICAL SLA RISK**).

---

## 🏗️ System Architecture

```mermaid
flowchart LR
    A[Logistics Manager / Operator] -->|Inputs Destination City & Order Specs| B(Streamlit Frontend Dashboard)
    B -->|Geocodes Coordinates & Transit Distance| C[(Geopy Nominatim Engine)]
    B -->|Fetches Arrival Date Satellite Forecast| D[(Open-Meteo Weather API)]
    B -->|Sends POST /predict JSON Payload| E(FastAPI Microservice Backend)
    E -->|Pydantic Schema Validation| F{Validation Gate}
    F -->|Feature Encoding & Reindexing| G[XGBoost Inference Engine]
    G -->|Probability & Delay Attribution| E
    E -->|Returns PredictResponse JSON| B
    B -->|Renders KPI Metrics, Root-Cause Alert & Trajectory| A



