import streamlit as st
import json
import os
from ultralytics import YOLO
from PIL import Image
import tempfile
import pandas as pd
from streamlit_js_eval import get_geolocation
from datetime import datetime
import pydeck as pdk
import math
import time
import streamlit.components.v1 as components

# ---------------- MODEL ---------------- #
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- CONSTANTS ---------------- #
LOW_AREA_THRESHOLD = 5000
MEDIUM_AREA_THRESHOLD = 20000
LOW_CONF_THRESHOLD = 0.5
JSON_FILE = "pothole_predictions.json"

# ---------------- DISTANCE ---------------- #
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ---------------- POPUP ALERT ---------------- #
def show_popup_alert(message):
    components.html(f"""
    <script>
    alert("{message}");
    </script>
    """, height=0)

# ---------------- GEOLOCATION ---------------- #
st.sidebar.header("📍 GPS Status")

location = get_geolocation()
lat, lon = None, None

if location and "coords" in location:
    lat = location["coords"].get("latitude")
    lon = location["coords"].get("longitude")
    st.sidebar.success(f"GPS: {lat:.5f}, {lon:.5f}")
else:
    st.sidebar.warning("⚠ Allow location access")

# ---------------- MODE ---------------- #
mode = st.sidebar.radio("Select Mode", ["Report", "Detect"])

# ---------------- SAVE FUNCTION ---------------- #
def process_and_save_prediction(image_path, latitude, longitude):
    results = model.predict(image_path, save=False, verbose=False)

    detections = []

    for r in results:
        if r.obb is not None:
            for box, conf in zip(r.obb.xyxy, r.obb.conf):
                x1, y1, x2, y2 = box.cpu().numpy()
                area = (x2 - x1) * (y2 - y1)

                if area < LOW_AREA_THRESHOLD and conf.item() < LOW_CONF_THRESHOLD:
                    severity = "LOW"
                elif area < MEDIUM_AREA_THRESHOLD:
                    severity = "MEDIUM"
                else:
                    severity = "HIGH"

                detections.append({
                    "severity": severity,
                    "confidence": float(conf.item())
                })

    if len(detections) == 0:
        return []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_data = {
        "latitude": latitude,
        "longitude": longitude,
        "timestamp": timestamp,
        "detections": detections
    }

    all_data = []

    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, "r") as f:
                all_data = json.load(f)
        except:
            all_data = []

    all_data.append(new_data)

    with open(JSON_FILE, "w") as f:
        json.dump(all_data, f, indent=4)

    return detections

# =========================================================
# 🟢 REPORT MODE
# =========================================================
if mode == "Report":

    st.title("🚧 Report Pothole")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:

        if lat is None or lon is None:
            st.error("❌ Location required")
            st.stop()

        image = Image.open(uploaded_file)
        st.image(image, width='stretch')

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name

        st.write("🔍 Detecting...")
        detections = process_and_save_prediction(temp_path, lat, lon)

        if len(detections) == 0:
            st.success("✅ No pothole → Not saved")
        else:
            st.error(f"⚠ {len(detections)} pothole(s) detected & saved")

# =========================================================
# 🔴 DETECT MODE
# =========================================================
elif mode == "Detect":

    st.title("📡 Live Detection")

    if lat is None or lon is None:
        st.error("❌ Location required")
        st.stop()

    st.success(f"📍 Your Location: {lat:.5f}, {lon:.5f}")

    # 🔥 SINGLE MAP PLACEHOLDER (FIX)
    map_placeholder = st.empty()

    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r") as f:
            data = json.load(f)

        nearby = []
        map_data = []

        # 🟢 USER
        map_data.append({
            "lat": lat,
            "lon": lon,
            "color": [0, 255, 0],
            "radius": 200,
            "type": "YOU"
        })

        for d in data:
            dlat = d.get("latitude")
            dlon = d.get("longitude")

            if dlat and dlon:
                dist = haversine(lat, lon, dlat, dlon)

                if dist <= 50:
                    nearby.append(dist)

                    map_data.append({
                        "lat": dlat,
                        "lon": dlon,
                        "color": [255, 0, 0],
                        "radius": 80,
                        "type": "POTHOLE"
                    })

        # 🚨 ALERT
        if "alert_sent" not in st.session_state:
            st.session_state.alert_sent = False

        if nearby and not st.session_state.alert_sent:
            st.error("🚨 Pothole ahead!")

            show_popup_alert(
                f"Pothole detected within {round(nearby[0],2)} meters!"
            )

            st.session_state.alert_sent = True

        if not nearby:
            st.session_state.alert_sent = False
            st.success("✅ No nearby potholes")

        # 🗺 MAP (ONLY ONE)
        with map_placeholder:
            df = pd.DataFrame(map_data)

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position='[lon, lat]',
                get_fill_color='color',
                get_radius='radius',
                pickable=True
            )

            st.pydeck_chart(pdk.Deck(
                layers=[layer],
                initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=15),
                tooltip={"html": "<b>{type}</b>"}
            ))

        # 🔄 Refresh
        time.sleep(3)
        st.rerun()

    else:
        st.warning("⚠ No pothole data yet")

# =========================================================
# 📊 TABLE VIEW
# =========================================================
st.subheader("📊 Pothole Data Table")

if os.path.exists(JSON_FILE):
    with open(JSON_FILE, "r") as f:
        data = json.load(f)

    rows = []

    for entry in data:
        for det in entry.get("detections", []):
            rows.append({
                "Latitude": entry["latitude"],
                "Longitude": entry["longitude"],
                "Timestamp": entry["timestamp"],
                "Severity": det["severity"],
                "Confidence": round(det["confidence"], 2)
            })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No pothole records yet")
else:
    st.info("No data file found")
