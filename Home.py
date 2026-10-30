# Streamlit_Accident_Analytics_App_Full.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, HeatMap
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import base64
import io
from datetime import datetime

# Optional ML libraries
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# -----------------------------
# App config & styling
# -----------------------------
st.set_page_config(page_title="🚦 Karnataka Road Safety Analytics", layout="wide")

st.markdown(
    """
    <style>
    .header-gradient {
        background: linear-gradient(90deg, #ff7a7a, #ffc86b, #ff86c3);
        padding: 18px; border-radius: 12px; color: white; font-weight:700;
    }
    .kpi {
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border-radius: 10px; padding: 14px; text-align:center;
    }
    .small {font-size:12px; color: #f3f3f3}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Simple login
# -----------------------------
USERNAME = "admin"
PASSWORD = "1234"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown('<div class="header-gradient">🚦 Karnataka Road Safety & Accident Analytics</div>', unsafe_allow_html=True)
    st.write("\n")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("✅ Login successful")
            st.experimental_rerun()  # Use new method st.experimental_rerun() for refresh
        else:
            st.error("❌ Invalid credentials")
    st.stop()

# -----------------------------
# Utility functions
# -----------------------------
@st.cache_data
def load_data(path="karnataka_accidents_2021_2025.csv"):
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.day_name()
    if "severity" in df.columns:
        df["severity_num"] = pd.to_numeric(df["severity"], errors="coerce").fillna(0).astype(int)
        severity_map = {1: "Minor", 2: "Moderate", 3: "Severe", 4: "Fatal"}
        df["severity_label"] = df["severity_num"].map(severity_map).fillna("Unknown")
    else:
        df["severity_label"] = "Unknown"
    for c in ["latitude", "longitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "city" not in df.columns:
        df["city"] = df.get("district", "Unknown")
    return df

def kpis_from_df(df):
    total = len(df)
    fatal = int(df["is_fatal"].sum()) if "is_fatal" in df.columns else int((df["severity_num"] == 4).sum())
    avg_cas = df["casualties"].mean() if "casualties" in df.columns else 0
    bad_weather = df[df["weather"].str.contains("rain|fog|storm|hail", case=False, na=False)]
    bad_weather_pct = (len(bad_weather)/total*100) if total>0 else 0
    return {"total": total, "fatal": fatal, "avg_cas": round(avg_cas,2), "bad_weather_pct": round(bad_weather_pct,1)}

def download_link(obj, filename, text):
    if isinstance(obj, pd.DataFrame):
        csv = obj.to_csv(index=False).encode("utf-8")
        b64 = base64.b64encode(csv).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    else:
        b64 = base64.b64encode(obj).decode()
        return f'<a href="data:file/octet-stream;base64,{b64}" download="{filename}">{text}</a>'

# -----------------------------
# Sidebar & Navigation
# -----------------------------
PAGES = ["Dashboard","Geospatial","City Insights","Comparisons","Reports","Forecasting","Admin"]
st.sidebar.title("📌 Navigation")
page = st.sidebar.selectbox("Choose page", PAGES)

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.query_params.clear()
    st.stop()

# -----------------------------
# Load data
# -----------------------------
DATA_PATH = "karnataka_accidents_2021_2025.csv"
if "uploaded_data" not in st.session_state:
    st.session_state["uploaded_data"] = None

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"]) 
if uploaded is not None:
    try:
        df_uploaded = pd.read_csv(uploaded)
        st.session_state["uploaded_data"] = df_uploaded
        st.sidebar.success("Uploaded")
    except:
        st.sidebar.error("Error reading CSV")

data = st.session_state["uploaded_data"] if st.session_state["uploaded_data"] is not None else load_data(DATA_PATH)

# -----------------------------
# Filters
# -----------------------------
st.sidebar.header("🔎 Filters")
years = sorted(data["year"].dropna().unique().astype(int).tolist()) if "year" in data.columns else []
selected_years = st.sidebar.multiselect("Select Year(s)", options=years, default=years)
severity_opts = sorted(data["severity_label"].dropna().unique().tolist()) if "severity_label" in data.columns else ["Unknown"]
selected_severity = st.sidebar.multiselect("Select Severity Level(s)", options=severity_opts, default=severity_opts)
weather_opts = sorted(data["weather"].dropna().unique().tolist()) if "weather" in data.columns else [""]
selected_weather = st.sidebar.multiselect("Select Weather", options=weather_opts, default=weather_opts)
road_opts = sorted(data["road_type"].dropna().unique().tolist()) if "road_type" in data.columns else [""]
selected_road = st.sidebar.multiselect("Select Road Type", options=road_opts, default=road_opts)
vehicle_opts = sorted(data["vehicle_type"].dropna().unique().tolist()) if "vehicle_type" in data.columns else [""]
selected_vehicle = st.sidebar.multiselect("Select Vehicle Type", options=vehicle_opts, default=vehicle_opts)

cities = ["All Karnataka"] + sorted(data["city"].dropna().unique().tolist()) if "city" in data.columns else ["All Karnataka"]
selected_city = st.sidebar.selectbox("Select City", cities)
map_mode = st.sidebar.radio("Map Visualization", options=["Markers","Clusters","Heatmap"], index=0)

# Apply filters
filtered = data.copy()
if selected_years and "year" in filtered.columns: filtered = filtered[filtered["year"].isin(selected_years)]
if selected_severity and "severity_label" in filtered.columns: filtered = filtered[filtered["severity_label"].isin(selected_severity)]
if selected_weather and "weather" in filtered.columns: filtered = filtered[filtered["weather"].isin(selected_weather)]
if selected_road and "road_type" in filtered.columns: filtered = filtered[filtered["road_type"].isin(selected_road)]
if selected_vehicle and "vehicle_type" in filtered.columns: filtered = filtered[filtered["vehicle_type"].isin(selected_vehicle)]
if selected_city != "All Karnataka": filtered = filtered[filtered["city"] == selected_city]

st.markdown('<div class="header-gradient">🚦 Karnataka Road Safety & Accident Analytics</div>', unsafe_allow_html=True)
st.write("\n")

# -----------------------------
# -----------------------------
# Dashboard page (enhanced)
# -----------------------------
if page == "Dashboard":
    st.subheader("📊 Dashboard Overview")
    kpis = kpis_from_df(filtered)
    k1,k2,k3,k4 = st.columns(4)
    k1.markdown(f"<div class='kpi'><h3>{kpis['total']:,}</h3><div class='small'>Total Accidents</div></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi'><h3>{kpis['fatal']:,}</h3><div class='small'>Fatal Accidents</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi'><h3>{kpis['avg_cas']}</h3><div class='small'>Avg Casualties</div></div>", unsafe_allow_html=True)
    k4.markdown(f"<div class='kpi'><h3>{kpis['bad_weather_pct']}%</h3><div class='small'>% in Bad Weather</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    # Accidents per Year
    if "year" in filtered.columns and not filtered.empty:
        yearly = filtered.groupby("year").size().reset_index(name="accidents")
        fig = px.bar(yearly, x="year", y="accidents", title="Accidents per Year", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # Severity Distribution
    if "severity_label" in filtered.columns:
        sev = filtered["severity_label"].value_counts().reset_index()
        sev.columns = ["severity","count"]
        fig2 = px.pie(sev, values="count", names="severity", hole=0.35)
        st.plotly_chart(fig2, use_container_width=True)

    # Severity vs Weather
    if "weather" in filtered.columns and "severity_label" in filtered.columns:
        weather_sev = filtered.groupby(["weather","severity_label"]).size().reset_index(name="count")
        fig3 = px.bar(weather_sev, x="weather", y="count", color="severity_label", barmode="stack", title="Accident Severity by Weather", template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)

    # Top 5 accident-prone cities
    if "city" in filtered.columns:
        top5 = filtered["city"].value_counts().head(5).reset_index()
        top5.columns = ["city","count"]
        st.bar_chart(top5.set_index("city"))

    # Top 10 accident-prone cities map
    if all(x in filtered.columns for x in ["city","latitude","longitude"]):
        top10_cities = filtered.groupby("city").agg({"latitude":"mean","longitude":"mean","severity_num":"mean","city":"count"}).rename(columns={"city":"count"}).sort_values("count", ascending=False).head(10).reset_index()
        if not top10_cities.empty:
            center_lat = top10_cities["latitude"].mean()
            center_lon = top10_cities["longitude"].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="CartoDB positron")
            for _, r in top10_cities.iterrows():
                color = "red" if r["severity_num"]>=3 else "orange" if r["severity_num"]>=2 else "green"
                folium.CircleMarker(location=[r["latitude"], r["longitude"]], radius=8, color=color, fill=True, fill_opacity=0.7,
                                    popup=f"{r['city']}<br>Avg Severity: {r['severity_num']:.1f}<br>Accidents: {r['count']}").add_to(m)
            st_folium(m, width=1000, height=500)

# -----------------------------
# Geospatial page
# -----------------------------
elif page=="Geospatial":
    st.subheader("🗺️ Geospatial Analysis")
    if filtered.empty or not all(x in filtered.columns for x in ["latitude","longitude"]):
        st.info("No geolocation data.")
    else:
        sample = filtered.sample(min(1500,len(filtered)), random_state=42)
        center_lat = sample["latitude"].mean()
        center_lon = sample["longitude"].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="CartoDB positron")
        if map_mode=="Markers":
            for _, r in sample.iterrows():
                color = "red" if r.get("severity_num",0)>=3 else "orange" if r.get("severity_num",0)==2 else "green"
                folium.CircleMarker([r["latitude"], r["longitude"]], radius=4,
                                    popup=f"{r.get('city','')} | Severity: {r.get('severity_label','')}", color=color, fill=True, fill_opacity=0.7).add_to(m)
        elif map_mode=="Clusters":
            cluster = MarkerCluster().add_to(m)
            for _, r in sample.iterrows():
                folium.Marker([r["latitude"], r["longitude"]], popup=f"{r.get('city','')} | Severity: {r.get('severity_label','')}").add_to(cluster)
        else:
            HeatMap(sample[["latitude","longitude"]].dropna().values.tolist(), radius=10, blur=15).add_to(m)
        st_folium(m, width=1000, height=600)

# -----------------------------
# City Insights
# -----------------------------
elif page=="City Insights":
    st.subheader("🏙️ City Insights")
    if selected_city=="All Karnataka":
        st.info("Select a city.")
    else:
        city_df = filtered.copy()
        st.write(f"**Total accidents:** {len(city_df):,}")
        st.write(f"**Average casualties:** {city_df['casualties'].mean() if 'casualties' in city_df.columns else 0:.2f}")
        st.write(f"**Fatal accidents:** {int((city_df['severity_num']==4).sum()) if 'severity_num' in city_df.columns else 0}")
        if not city_df.empty and "year" in city_df.columns:
            st.bar_chart(city_df.groupby("year").size())

        # Top factors
        st.markdown("#### Top Contributing Factors")
        cols = st.columns(3)
        with cols[0]:
            st.markdown("**Weather**")
            for w,c in city_df["weather"].value_counts().head(5).items():
                st.write(f"- {w}: {c}")
        with cols[1]:
            st.markdown("**Road Types**")
            for r,c in city_df["road_type"].value_counts().head(5).items():
                st.write(f"- {r}: {c}")
        with cols[2]:
            st.markdown("**Vehicle Types**")
            for v,c in city_df["vehicle_type"].value_counts().head(5).items():
                st.write(f"- {v}: {c}")

        # Suggested measures
        st.markdown("#### Suggested Measures")
        measures = []
        top_weather = city_df["weather"].value_counts().head(3).index.tolist()
        top_road = city_df["road_type"].value_counts().head(3).index.tolist()
        top_vehicle = city_df["vehicle_type"].value_counts().head(3).index.tolist()
        if any("rain" in str(w).lower() or "fog" in str(w).lower() for w in top_weather):
            measures.append("Improve drainage, add visibility signage, temporary speed limits.")
        if any("Highway" in str(r) for r in top_road):
            measures.append("Increase highway enforcement and layby rest stops.")
        if any("Two-Wheeler" in str(v) for v in top_vehicle):
            measures.append("Helmet law enforcement and rider safety drives.")
        if not measures:
            measures.append("General awareness recommended.")
        for m in measures:
            st.write(f"- {m}")

# -----------------------------
# Comparisons Page
# -----------------------------
elif page == "Comparisons":
    st.subheader("📈 Top 5 Accident-Prone Cities Comparison")
    if "city" in filtered.columns:
        city_counts = filtered["city"].value_counts().head(5).reset_index()
        city_counts.columns = ["city", "count"]
        fig = px.bar(city_counts, x="city", y="count", color="city", title="Top 5 Cities by Accidents", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
        # Severity comparison
        if "severity_label" in filtered.columns:
            sev_cmp = filtered[filtered["city"].isin(city_counts["city"])].groupby(["city","severity_label"]).size().reset_index(name="count")
            fig2 = px.bar(sev_cmp, x="city", y="count", color="severity_label", barmode="stack", title="Severity Distribution for Top 5 Cities", template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Reports Page
# -----------------------------
elif page == "Reports":
    st.subheader("📄 Generate Reports")
    st.markdown("Download filtered data or generate a summary PDF report.")
    
    # CSV download
    csv_link = download_link(filtered, "filtered_accidents.csv", "📥 Download CSV")
    st.markdown(csv_link, unsafe_allow_html=True)
    
    # PDF report
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elems = [Paragraph("Karnataka Road Safety Report", styles['Title'])]
    elems.append(Spacer(1,12))
    elems.append(Paragraph(f"Total Accidents: {len(filtered):,}", styles['Normal']))
    elems.append(Paragraph(f"Fatal Accidents: {int((filtered['severity_num']==4).sum()) if 'severity_num' in filtered.columns else 0}", styles['Normal']))
    elems.append(Paragraph(f"Average Casualties: {filtered['casualties'].mean() if 'casualties' in filtered.columns else 0:.2f}", styles['Normal']))
    elems.append(Spacer(1,12))
    elems.append(Paragraph("Top 5 Cities by Accidents:", styles['Heading2']))
    if "city" in filtered.columns:
        for c, v in filtered["city"].value_counts().head(5).items():
            elems.append(Paragraph(f"{c}: {v}", styles['Normal']))
    doc.build(elems)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    pdf_link = download_link(pdf_bytes, "report.pdf", "📥 Download PDF Report")
    st.markdown(pdf_link, unsafe_allow_html=True)

# -----------------------------
# Forecasting Page
# -----------------------------
elif page == "Forecasting":
    st.subheader("🔮 Accident Forecasting")
    if "date" in filtered.columns:
        ts_data = filtered.groupby("date").size().reset_index(name="accidents")
        ts_data = ts_data.sort_values("date")
        st.line_chart(ts_data.set_index("date")["accidents"], width=900, height=400)
        
        if PROPHET_AVAILABLE:
            st.write("Using Prophet to forecast next 30 days...")
            df_prophet = ts_data.rename(columns={"date":"ds","accidents":"y"})
            m = Prophet(daily_seasonality=True)
            m.fit(df_prophet)
            future = m.make_future_dataframe(periods=30)
            forecast = m.predict(future)
            fig = px.line(forecast, x="ds", y="yhat", title="Forecasted Accidents (Next 30 Days)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Prophet not installed, using simple moving average as fallback.")
            ts_data["MA_7"] = ts_data["accidents"].rolling(7).mean()
            st.line_chart(ts_data.set_index("date")[["accidents","MA_7"]], width=900, height=400)

# -----------------------------
# Admin Page
# -----------------------------
elif page == "Admin":
    st.subheader("🛠️ Admin Panel")
    st.markdown("Upload a new CSV dataset to update the dashboard data.")
    uploaded_new = st.file_uploader("Upload CSV", type=["csv"], key="admin_upload")
    if uploaded_new is not None:
        try:
            df_new = pd.read_csv(uploaded_new)
            st.session_state["uploaded_data"] = df_new
            st.success("✅ New dataset uploaded successfully. Dashboard updated!")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"❌ Error reading CSV: {e}")
