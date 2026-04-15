import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextlib
import io
from wqchartpy import (gibbs, triangle_piper, rectangle_piper, durvo, 
                       hfed, stiff, chadha, gaillardet, schoeller, chernoff)

# 1. Config & Standards
st.set_page_config(page_title="Professional Water Suitability Suite", layout="wide")

WHO_STD = {"pH":8.5, "Ca":75, "Mg":50, "Na":200, "K":12, "HCO3":300, "Cl":250, "SO4":250, "TDS":500}
WEIGHTS = {"pH":4, "Ca":2, "Mg":2, "Na":3, "K":1, "HCO3":2, "Cl":3, "SO4":3, "TDS":4}
EQ_WT = {"Ca":20.04, "Mg":12.15, "Na":23, "K":39.1, "HCO3":61, "CO3":30}

# 2. Logic Functions
def process_data(df):
    total_wt = sum(WEIGHTS.values())
    rel_wt = {k: v/total_wt for k, v in WEIGHTS.items()}
    
    # Calculate Drinking WQI
    df["WQI_Drinking"] = df.apply(lambda row: sum((row[p]/WHO_STD[p]*100)*rel_wt[p] for p in WHO_STD if p in row), axis=1)
    
    # Unit Conversion to meq/L for Irrigation Indices
    for ion, ew in EQ_WT.items():
        if ion in df.columns:
            df[f"{ion}_meq"] = df[ion]/ew
            
    # Irrigation Indices Logic
    df["SAR"] = df["Na_meq"]/np.sqrt((df["Ca_meq"] + df["Mg_meq"])/2)
    df["Na_percent"] = (df["Na_meq"]+df["K_meq"]) / (df["Ca_meq"]+df["Mg_meq"]+df["Na_meq"]+df["K_meq"]) * 100
    df["RSC"] = (df.get("HCO3_meq", 0) + df.get("CO3_meq", 0)) - (df["Ca_meq"] + df["Mg_meq"])
    df["Kelly_Ratio"] = df["Na_meq"] / (df["Ca_meq"] + df["Mg_meq"])
    return df

def get_suitability_report(wqi, sar, rsc, kelly):
    is_drinkable = wqi <= 100
    is_irrigation = sar < 18 and rsc < 2.5 and kelly < 1
    
    if is_drinkable and is_irrigation:
        return "✅ Highly Suitable: Safe for Drinking & Irrigation", "green"
    elif is_drinkable:
        return "⚠️ Suitable for Drinking Only: Potential Soil Hazard", "blue"
    elif is_irrigation:
        return "🚜 Suitable for Irrigation Only: Exceeds Drinking Safety Limits", "orange"
    else:
        return "🚨 Not Suitable: Unsafe for Human or Agricultural use", "red"

# 3. Sidebar: Input Configuration
st.sidebar.title("Data Management")
input_mode = st.sidebar.radio("Input Method:", ["Single Sample (Manual)", "Batch Analysis (Upload CSV)"])

if input_mode == "Single Sample (Manual)":
    st.sidebar.subheader("Chemical Parameters (mg/L)")
    ph = st.sidebar.number_input("pH", 7.8)
    tds = st.sidebar.number_input("TDS", 233.0)
    ca = st.sidebar.number_input("Ca", 32.0)
    mg = st.sidebar.number_input("Mg", 6.0)
    na = st.sidebar.number_input("Na", 28.0)
    k = st.sidebar.number_input("K", 2.8)
    hco3 = st.sidebar.number_input("HCO3", 73.0)
    cl = st.sidebar.number_input("Cl", 43.0)
    so4 = st.sidebar.number_input("SO4", 48.0)
    
    data = {
        "Sample": ["User_1"], "Label": ["Manual"], "Color": ["blue"], "Marker": ["o"], "Size": [50], "Alpha": [0.8],
        "pH": [ph], "Ca": [ca], "Mg": [mg], "Na": [na], "K": [k], "HCO3": [hco3], "CO3": [0], "Cl": [cl], "SO4": [so4], "TDS": [tds]
    }
    df = pd.DataFrame(data)
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Fill missing plotting columns if necessary
        for col, val in [("Color","red"), ("Marker","o"), ("Size",50), ("Alpha",0.8), ("Label","CSV")]:
            if col not in df.columns: df[col] = val
    else:
        st.info("Please upload a CSV file with headers: Sample, pH, Ca, Mg, Na, K, HCO3, Cl, SO4, TDS")
        st.stop()

# 4. Data Execution
df = process_data(df)
first_sample = df.iloc[0]

# 5. UI Dashboard
st.title("🌊 Water Quality Suitability Dashboard")

# Top Level Verdict Bar
verdict_text, verdict_color = get_suitability_report(
    first_sample["WQI_Drinking"], first_sample["SAR"], first_sample["RSC"], first_sample["Kelly_Ratio"]
)
st.markdown(f"""<div style="background-color:{verdict_color}; padding:20px; border-radius:10px; color:white; font-size:24px; font-weight:bold; text-align:center;">
            {verdict_text}</div>""", unsafe_allow_html=True)
st.divider()

# Result Columns
tab1, tab2, tab3 = st.tabs(["📊 Analysis Results", "📈 Hydrochemical Plots", "📑 Dataset View"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Drinking Quality")
        st.metric("Drinking WQI", f"{first_sample['WQI_Drinking']:.2f}")
        st.write(f"Class: **{'Excellent' if first_sample['WQI_Drinking'] < 50 else 'Good' if first_sample['WQI_Drinking'] < 100 else 'Unsuitable'}**")
    with c2:
        st.subheader("Irrigation Quality")
        st.metric("SAR Index", f"{first_sample['SAR']:.2f}")
        st.write(f"Sodium %: **{first_sample['Na_percent']:.2f}%**")

with tab2:
    st.subheader("Hydrochemical Diagrams")
    plot_option = st.selectbox("Select Diagram Type:", 
        ["Gibbs", "Triangle Piper", "Rectangle Piper", "Durov", "HFE-D", "Stiff", "Chadha", "Gaillardet", "Schoeller", "Chernoff"])
    
    plot_map = {
        "Gibbs": gibbs, "Triangle Piper": triangle_piper, "Rectangle Piper": rectangle_piper,
        "Durov": durvo, "HFE-D": hfed, "Stiff": stiff, "Chadha": chadha,
        "Gaillardet": gaillardet, "Schoeller": schoeller, "Chernoff": chernoff
    }

    if st.button("Generate Selected Plot"):
        # contextlib prevents the app from crashing on 'I/O Error 5'
        with contextlib.redirect_stdout(io.StringIO()):
            plot_func = plot_map[plot_option]
            plot_func.plot(df, unit='mg/L', figname="water_plot", figformat='png')
        st.image("water_plot.png", use_container_width=True)

with tab3:
    st.write("Full Processed Data (including meq/L conversions):")
    st.dataframe(df)
    st.download_button("Download Results (CSV)", df.to_csv(index=False), "water_suitability_results.csv", "text/csv")

# 6. Documentation
with st.expander("📖 Scientific Definitions"):
    st.markdown("""
    - **WQI**: Water Quality Index (WHO standards). Values < 100 are drinkable.
    - **SAR**: Sodium Adsorption Ratio. Values < 10 are excellent for irrigation.
    - **RSC**: Residual Sodium Carbonate. Values > 2.5 are hazardous for soil structure.
    - **Kelly Ratio**: Sodium divided by Calcium and Magnesium. Values > 1 indicate excess sodium.
    """)
