from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re
import joblib


# ============================================
# CREATE API
# ============================================

app = FastAPI(
    title="AI Disaster Response API",
    description="AI-powered disaster response and rescue coordinator",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "https://resqai-command-center.netlify.app"
    ],
)


# ============================================
# LOAD DATA
# ============================================

reports = pd.read_csv(
    "data/emergency_reports.csv"
)

teams = pd.read_csv(
    "data/rescue_teams.csv"
)

hospitals = pd.read_csv(
    "data/hospitals.csv"
)

shelters = pd.read_csv(
    "data/shelters.csv"
)


# ============================================
# LOAD ML MODELS
# ============================================

disaster_model = joblib.load(
    "models/disaster_type_model.joblib"
)

disaster_vectorizer = joblib.load(
    "models/disaster_vectorizer.joblib"
)

severity_model = joblib.load(
    "models/severity_model.joblib"
)

severity_vectorizer = joblib.load(
    "models/severity_vectorizer.joblib"
)


# ============================================
# REQUEST FORMAT
# ============================================

class EmergencyReport(BaseModel):

    text: str
    latitude: float
    longitude: float


# ============================================
# HAVERSINE DISTANCE
# ============================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(
        np.sqrt(a)
    )

    return R * c


# ============================================
# RESOURCE MATCHING
# ============================================

def resource_match(
    resource,
    equipment
):

    mapping = {
        "rescue boat": "boat",
        "ambulance": "ambulance",
        "rescue vehicle": "rescue_vehicle",
        "heavy equipment": "heavy_equipment"
    }

    required = mapping.get(
        resource,
        resource
    )

    equipment_list = [
        item.strip().lower()
        for item in equipment.split(",")
    ]

    return required.lower() in equipment_list


# ============================================
# ROOT ENDPOINT
# ============================================

@app.get("/")
def home():

    return {
        "message": "AI Disaster Response API is running",
        "status": "online"
    }


# ============================================
# ANALYZE EMERGENCY
# ============================================

@app.post("/analyze")
def analyze_emergency(
    report: EmergencyReport
):

    text = report.text


    # ========================================
    # DISASTER TYPE
    # ========================================

    disaster_vector = (
        disaster_vectorizer.transform([text])
    )

    predicted_disaster = (
        disaster_model.predict(
            disaster_vector
        )[0]
    )


    # ========================================
    # SEVERITY
    # ========================================

    severity_vector = (
        severity_vectorizer.transform([text])
    )

    predicted_severity = (
        severity_model.predict(
            severity_vector
        )[0]
    )


    # ========================================
    # PEOPLE AFFECTED
    # ========================================

    people_match = re.search(
        r"(\d+)\s*(?:people|persons|person|individuals)",
        text.lower()
    )

    if people_match:

        people_affected = int(
            people_match.group(1)
        )

    else:

        people_affected = 0


    # ========================================
    # TRAPPED
    # ========================================

    trapped_keywords = [
        "trapped",
        "stranded",
        "stuck",
        "cannot escape"
    ]

    people_trapped = any(
        word in text.lower()
        for word in trapped_keywords
    )


    # ========================================
    # MEDICAL
    # ========================================

    medical_keywords = [
        "injured",
        "injury",
        "medical",
        "ambulance",
        "hospital",
        "bleeding"
    ]

    medical_required = any(
        word in text.lower()
        for word in medical_keywords
    )


    # ========================================
    # PRIORITY
    # ========================================

    priority_score = 0

    if predicted_severity == "critical":

        priority_score += 50

    elif predicted_severity == "high":

        priority_score += 30

    elif predicted_severity == "moderate":

        priority_score += 15


    priority_score += min(
        people_affected * 0.5,
        25
    )


    if people_trapped:

        priority_score += 15


    if medical_required:

        priority_score += 10


    priority_score = min(
        priority_score,
        100
    )


    if priority_score >= 70:

        priority_level = "P1 - CRITICAL"

    elif priority_score >= 40:

        priority_level = "P2 - HIGH"

    else:

        priority_level = "P3 - MODERATE"


    # ========================================
    # REQUIRED RESOURCES
    # ========================================

    required_resources = []


    if predicted_disaster == "flood":

        required_resources.append(
            "rescue boat"
        )

    elif predicted_disaster == "landslide":

        required_resources.append(
            "heavy equipment"
        )

    elif predicted_disaster == "earthquake":

        required_resources.append(
            "search and rescue equipment"
        )

    elif predicted_disaster == "cloudburst":

        required_resources.append(
            "rescue vehicle"
        )


    if people_trapped:

        required_resources.append(
            "rescue team"
        )


    if medical_required:

        required_resources.append(
            "ambulance"
        )


    if people_affected >= 20:

        required_resources.append(
            "additional rescue team"
        )


    if predicted_severity == "critical":

        required_resources.append(
            "emergency response unit"
        )


    # ========================================
    # RESCUE TEAM ASSIGNMENT
    # ========================================

    recommendations = []


    for _, team in teams.iterrows():

        available = team["available"]

        if isinstance(
            available,
            str
        ):

            available = (
                available.lower()
                in ["true", "yes", "1"]
            )

        else:

            available = bool(available)


        if not available:

            continue


        distance = haversine_distance(
            report.latitude,
            report.longitude,
            team["latitude"],
            team["longitude"]
        )


        equipment_matches = sum(
            resource_match(
                resource,
                team["equipment"]
            )
            for resource in required_resources
        )


        score = (
            distance
            -
            equipment_matches * 5
        )


        recommendations.append({

            "team_id": str(
                team["team_id"]
            ),

            "distance_km": round(
                distance,
                2
            ),

            "equipment": str(
                team["equipment"]
            ),

            "equipment_matches":
                int(equipment_matches),

            "assignment_score":
                round(score, 2)
        })


    recommendations.sort(
        key=lambda x:
        x["assignment_score"]
    )


    best_team = (
        recommendations[0]
        if recommendations
        else None
    )


    # ========================================
    # HOSPITAL ASSIGNMENT
    # ========================================

    hospital_recommendations = []


    for _, hospital in hospitals.iterrows():

        if hospital["beds_available"] <= 0:

            continue


        distance = haversine_distance(
            report.latitude,
            report.longitude,
            hospital["latitude"],
            hospital["longitude"]
        )


        hospital_recommendations.append({

            "hospital_id": str(
                hospital["hospital_id"]
            ),

            "hospital_name": str(
                hospital["name"]
            ),

            "beds_available":
                int(
                    hospital["beds_available"]
                ),

            "distance_km":
                round(
                    distance,
                    2
                )
        })


    hospital_recommendations.sort(
        key=lambda x:
        x["distance_km"]
    )


    best_hospital = (
        hospital_recommendations[0]
        if hospital_recommendations
        else None
    )


    # ========================================
    # FINAL RESPONSE
    # ========================================

    return {

        "incident": {

            "report": text,

            "latitude":
                report.latitude,

            "longitude":
                report.longitude
        },

        "ai_analysis": {

            "disaster_type":
                predicted_disaster,

            "severity":
                predicted_severity,

            "people_affected":
                people_affected,

            "people_trapped":
                people_trapped,

            "medical_required":
                medical_required
        },

        "priority": {

            "score":
                priority_score,

            "level":
                priority_level
        },

        "resources":
            required_resources,

        "rescue_assignment":
            best_team,

        "medical_response":
            best_hospital
    }