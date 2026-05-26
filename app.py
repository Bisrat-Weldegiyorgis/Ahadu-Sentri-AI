from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from collections import deque
import pandas as pd
import joblib
import random
import threading
import time

app = FastAPI(title="Ahadu IDS (CIC-IDS Based)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# =====================================================
# LOAD MODEL + FEATURES
# =====================================================

try:
    model = joblib.load("ids_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    MODEL_LOADED = True
    print("✅ Model and features loaded")
except Exception as e:
    print("❌ Load error:", e)
    model = None
    feature_names = []
    MODEL_LOADED = False

# =====================================================
# LIVE STORAGE
# =====================================================

live_logs = deque(maxlen=100)

# =====================================================
# ROOT
# =====================================================

@app.get("/")
def root():
    return {"status": "Ahadu IDS running"}

# =====================================================
# HEALTH
# =====================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "features": len(feature_names)
    }

# =====================================================
# SAFE PREDICTION FUNCTION (KEY FIX)
# =====================================================

def safe_predict(df):

    # 🔥 FORCE ALIGN TO TRAINING FEATURES
    df = df.reindex(columns=feature_names, fill_value=0)

    pred = model.predict(df)[0]

    confidence = 1.0
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(df).max())

    return pred, confidence

# =====================================================
# PREDICT API
# =====================================================

@app.post("/predict/")
def predict(payload: dict):

    if not MODEL_LOADED:
        raise HTTPException(500, "Model not loaded")

    try:
        df = pd.DataFrame([payload])

        pred, conf = safe_predict(df)

        result = {
            "prediction": "attack" if pred == 1 else "normal",
            "confidence": round(conf, 4)
        }

        live_logs.appendleft(result)

        return result

    except Exception as e:
        raise HTTPException(400, str(e))

# =====================================================
# LIVE GENERATOR (REALISTIC CIC-IDS STYLE)
# =====================================================

def generate_live():

    while True:
        try:

            # 🔥 create empty feature set
            sample = {col: 0 for col in feature_names}

            # inject random activity into some features
            random_cols = random.sample(feature_names, min(15, len(feature_names)))

            for col in random_cols:
                sample[col] = random.randint(0, 1000)

            df = pd.DataFrame([sample])

            pred, conf = safe_predict(df)

            result = {
                "prediction": "attack" if pred == 1 else "normal",
                "confidence": round(conf, 4)
            }

            live_logs.appendleft(result)

            time.sleep(2)

        except Exception as e:
            print("Live generator error:", e)
            time.sleep(2)

# =====================================================
# START BACKGROUND THREAD
# =====================================================

@app.on_event("startup")
def startup():
    print("🚀 Live IDS generator started")
    thread = threading.Thread(target=generate_live, daemon=True)
    thread.start()

# =====================================================
# LIVE API
# =====================================================

@app.get("/live")
def live():
    return list(live_logs)

# =====================================================
# YOUR ORIGINAL DASHBOARD (UNCHANGED)
# =====================================================

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html>

<head>

<title>Ahadu IDS Dashboard</title>

<link rel="stylesheet"
href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<style>

body{
    margin:0;
    font-family:Arial;
    background:#0b1220;
    color:white;
}

header{
    background:#111827;
    padding:15px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
    color:#38bdf8;
}

.container{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:15px;
    padding:15px;
}

.card{
    background:#111827;
    padding:15px;
    border-radius:10px;
}

.stats{
    display:flex;
    gap:10px;
    margin-bottom:15px;
}

.stat{
    flex:1;
    background:#0f172a;
    padding:10px;
    border-radius:8px;
    text-align:center;
}

.logs{
    height:420px;
    overflow-y:auto;
}

.log{
    padding:10px;
    margin-bottom:10px;
    border-radius:6px;
}

.normal{
    background:rgba(34,197,94,0.15);
    border-left:4px solid #22c55e;
}

.attack{
    background:rgba(239,68,68,0.15);
    border-left:4px solid #ef4444;
}

#map{
    height:420px;
    border-radius:10px;
}

</style>

</head>

<body>

<header>
🛡 AHADU AI IDS DASHBOARD
</header>

<div class="container">

    <div class="card">

        <h3>🚨 Live Threat Logs</h3>

        <div class="stats">

            <div class="stat">
                <div id="total">0</div>
                <small>Total</small>
            </div>

            <div class="stat">
                <div id="attacks">0</div>
                <small>Attacks</small>
            </div>

            <div class="stat">
                <div id="normal">0</div>
                <small>Normal</small>
            </div>

        </div>

        <div class="logs" id="logs"></div>

    </div>

    <div class="card">

        <h3>🌍 Global Threat Map</h3>

        <div id="map"></div>

    </div>

</div>

<script>

let map = L.map('map').setView([20,0],2);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution:'OpenStreetMap'
}).addTo(map);

async function updateDashboard(){

    const res = await fetch('/live');
    const data = await res.json();

    let logs = document.getElementById("logs");
    logs.innerHTML = "";

    let total = data.length;
    let attacks = 0;
    let normal = 0;

    data.forEach(item => {

        let cls = item.prediction === "normal" ? "normal" : "attack";

        if(item.prediction === "normal") normal++;
        else attacks++;

        logs.innerHTML += `
            <div class="log ${cls}">
                <b>${item.prediction.toUpperCase()}</b><br>
                Confidence: ${item.confidence}
            </div>
        `;

        if(item.prediction !== "normal"){
            L.circleMarker([10 + Math.random()*30, 10 + Math.random()*60], {
                radius: 8,
                color: 'red'
            }).addTo(map)
            .bindPopup("🚨 Attack detected");
        }
    });

    document.getElementById("total").innerText = total;
    document.getElementById("attacks").innerText = attacks;
    document.getElementById("normal").innerText = normal;
}

setInterval(updateDashboard, 2000);
updateDashboard();

</script>

</body>
</html>
"""
