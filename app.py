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
    # Logic for Verdict
    is_drinkable = wqi <= 100
    is_irrigation = sar < 18 and rsc < 2.5 and kelly < 1
    if is_drinkable and is_irrigation: return "✅ Highly Suitable: Safe for Drinking & Irrigation", "green"
    elif is_drinkable: return "⚠️ Suitable for Drinking Only: High Mineral Hazard for Soil", "blue"
    elif is_irrigation: return "🚜 Suitable for Irrigation Only: Exceeds Drinking Safety Limits", "orange"
    else: return "🚨 Not Suitable for Human or Agricultural use", "red"

# 3. Sidebar: Input Configuration
st.sidebar.title("Data Source")
input_mode = st.sidebar.radio("Choose Input:", ["Manual Entry", "Upload CSV"])

if input_mode == "Manual Entry":
    st.sidebar.subheader("Chemical Parameters (mg/L)")
    ph = st.sidebar.number_input("pH", 7.8)
    tds = st.sidebar.number_input("TDS", 233.0)
    ca = st.sidebar.number_input("Ca", 32.0), st.sidebar.number_input("Mg", 6.0), st.sidebar.number_input("Na", 28.0)
    k, hco3, cl, so4 = st.sidebar.number_input("K", 2.8), st.sidebar.number_input("HCO3", 73.0), st.sidebar.number_input("Cl", 43.0), st.sidebar.number_input("SO4", 48.0)
    
    data = {"Sample": ["User"], "Label": ["Manual"], "Color": ["blue"], "Marker": ["o"], "Size": [50], "Alpha": [0.8],
            "pH": [ph], "Ca": [ca], "Mg": [ca], "Na": [ca], "K": [k], "HCO3": [hco3], "CO3": [0], "Cl": [cl], "SO4": [so4], "TDS": [tds]}
    df = process_data(pd.DataFrame(data))
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Ensure standard visual columns for plotting
        for col, val in [("Color","red"), ("Marker","o"), ("Size",50), ("Alpha",0.8), ("Label","CSV")]:
            if col not in df.columns: df[col] = val
    else:
        st.info("Please upload a CSV file with headers: Sample, pH, Ca, Mg, Na, K, HCO3, Cl, SO4, TDS")
        st.stop()
        
# 4. Processing (needed to get the verdict first)
df = process_data(df)
wqi, sar, rsc, kelly = df["WQI_Drinking"].iloc[0], df["SAR"].iloc[0], df["RSC"].iloc[0], df["Kelly_Ratio"].iloc[0]

# 5. UI Layout
st.title("🌊 Professional Water Chemistry Suite")

# Top Verdict Bar
verdict, color = get_suitability_report(wqi, sar, rsc, kelly)
st.markdown(f"""<div style="background-color:{color}; padding:20px; border-radius:10px; color:white; font-size:24px; font-weight:bold; text-align:center;">
            {verdict}</div>""", unsafe_allow_html=True)
st.divider()

# TAB SECTION
tab1, tab2, tab3 = st.tabs(["📊 Suitability Summary", "📈 Hydrochemical Plots", "📑 Raw Data"])

with tab1:
    # 📖 --- Scientific Definitions: Up Front & Explaining all indices ---
    st.info("📖 ### Hydrochemical Guide & Scientific Definitions")
    guide1, guide2 = st.columns(2)
    with guide1:
        st.markdown("""
        **Drinking Quality:**
        * **WQI** (Water Quality Index): General drinking safety. Values **< 100** are drinkable. Classifications are based on total minerals.
        """)
    with guide2:
        st.markdown("""
        **Irrigation Suitability:**
        * **SAR** (Sodium Adsorption Ratio): Measures alkali hazard. Values **< 10** are excellent.
        * **Na%**: Percent of sodium vs. other cations. Over **60%** is hazardous.
        * **RSC** (Residual Sodium Carbonate): Hazard of carbonates. Values **< 1.25** are safe.
        * **Kelly Ratio**: Excess sodium hazard. Values **> 1** can affect soil permeability.
        """)
    st.divider()
    
    # 🧪 --- Analysis Results with Crisp Icons ---
    result1, result2 = st.columns(2)
    
    with result1:
        # Added 🚰 crisp image/emoji
        st.subheader("🚰 Drinking Quality")
        st.metric("Drinking WQI", f"{wqi:.2f}")
        wqi_class = 'Excellent' if wqi < 50 else 'Good' if wqi < 100 else 'Poor' if wqi < 200 else 'Unsuitable'
        st.write(f"Class: **{wqi_class}**")
    
    with result2:
        # Added 🌾 crisp image/emoji
        st.subheader("🌾 Irrigation Suitability")
        st.metric("SAR Index", f"{sar:.2f}")
        st.write(f"**Irrigation Status:** {'Excellent (<10)' if sar < 10 else 'Hazardous (>26)' if sar > 26 else 'Good'}")

with tab2:
    st.subheader("Generate Specialized Hydrochemical Plots")
    plot_option = st.selectbox("Select Diagram:", 
        ["Gibbs", "Triangle Piper", "Rectangle Piper", "Durov", "HFE-D", "Stiff", "Chadha", "Gaillardet", "Schoeller", "Chernoff"])
    
    plot_map = {
        "Gibbs": gibbs, "Triangle Piper": triangle_piper, "Rectangle Piper": rectangle_piper,
        "Durov": durvo, "HFE-D": hfed, "Stiff": stiff, "Chadha": chadha,
        "Gaillardet": gaillardet, "Schoeller": schoeller, "Chernoff": chernoff
    }

    if st.button("Generate selected Diagram"):
        with contextlib.redirect_stdout(io.StringIO()):
            plot_func = plot_map[plot_option]
            plot_func.plot(df, unit='mg/L', figname="current_plot", figformat='png')
        st.image("current_plot.png", use_container_width=True)

with tab3:
    st.dataframe(df)
    st.download_button("Download Processed Data", df.to_csv(index=False), "water_results.csv", "text/csv")
