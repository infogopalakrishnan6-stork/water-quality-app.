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
    
    # Drinking WQI Calculation
    # We use .get() to avoid errors if a column is missing
    df["WQI_Drinking"] = df.apply(lambda row: sum((row[p]/WHO_STD[p]*100)*rel_wt[p] for p in WHO_STD if p in row), axis=1)
    
    # Unit Conversion to meq/L
    for ion, ew in EQ_WT.items():
        if ion in df.columns:
            df[f"{ion}_meq"] = df[ion]/ew
            
    # Irrigation Indices
    df["SAR"] = df["Na_meq"]/np.sqrt((df["Ca_meq"] + df["Mg_meq"])/2)
    df["Na_percent"] = (df["Na_meq"]+df["K_meq"]) / (df["Ca_meq"]+df["Mg_meq"]+df["Na_meq"]+df["K_meq"]) * 100
    df["RSC"] = (df.get("HCO3_meq", 0) + df.get("CO3_meq", 0)) - (df["Ca_meq"] + df["Mg_meq"])
    df["Kelly_Ratio"] = df["Na_meq"] / (df["Ca_meq"] + df["Mg_meq"])
    return df

def get_suitability_report(wqi, sar, rsc, kelly):
    is_drinkable = wqi <= 100
    is_irrigation = sar < 18 and rsc < 2.5 and kelly < 1
    if is_drinkable and is_irrigation: return "✅ Highly Suitable: Safe for Drinking & Irrigation", "green"
    elif is_drinkable: return "⚠️ Suitable for Drinking Only", "blue"
    elif is_irrigation: return "🚜 Suitable for Irrigation Only", "orange"
    else: return "🚨 Not Suitable for use", "red"

# 3. Sidebar: Input Configuration
st.sidebar.title("Data Source")
input_mode = st.sidebar.radio("Choose Input:", ["Manual Entry", "Upload CSV"])

if input_mode == "Manual Entry":
    st.sidebar.subheader("Chemical Parameters (mg/L)")
    ph = st.sidebar.number_input("pH", value=7.8)
    tds = st.sidebar.number_input("TDS", value=233.0)
    ca = st.sidebar.number_input("Ca", value=32.0)
    mg = st.sidebar.number_input("Mg", value=6.0)
    na = st.sidebar.number_input("Na", value=28.0)
    k = st.sidebar.number_input("K", value=2.8)
    hco3 = st.sidebar.number_input("HCO3", value=73.0)
    cl = st.sidebar.number_input("Cl", value=43.0)
    so4 = st.sidebar.number_input("SO4", value=48.0)
    
    data = {
        "Sample": ["User"], "Label": ["Manual"], "Color": ["blue"], "Marker": ["o"], "Size": [50], "Alpha": [0.8],
        "pH": ph, "Ca": ca, "Mg": mg, "Na": na, "K": k, "HCO3": hco3, "CO3": 0, "Cl": cl, "SO4": so4, "TDS": tds
    }
    df = pd.DataFrame(data)
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        for col, val in [("Color","red"), ("Marker","o"), ("Size",50), ("Alpha",0.8), ("Label","CSV")]:
            if col not in df.columns: df[col] = val
    else:
        st.info("Please upload a CSV file to proceed.")
        st.stop()

# 4. Processing
df = process_data(df)
wqi, sar, rsc, kelly = df["WQI_Drinking"].iloc[0], df["SAR"].iloc[0], df["RSC"].iloc[0], df["Kelly_Ratio"].iloc[0]

# 5. UI Layout
st.title("🌊 Professional Water Chemistry Suite")

verdict, color = get_suitability_report(wqi, sar, rsc, kelly)
st.markdown(f"""<div style="background-color:{color}; padding:20px; border-radius:10px; color:white; font-size:24px; font-weight:bold; text-align:center;">
            {verdict}</div>""", unsafe_allow_html=True)
st.divider()

tab1, tab2, tab3 = st.tabs(["📊 Suitability Summary", "📈 Hydrochemical Plots", "📑 Raw Data"])

with tab1:
    st.info("📖 ### Hydrochemical Guide & Scientific Definitions")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Drinking Quality:** WQI < 100 is safe.")
    with g2:
        st.markdown("**Irrigation:** SAR < 10, RSC < 1.25, and Kelly < 1 are safe.")
    st.divider()
    
    res1, res2 = st.columns(2)
    with res1:
        st.subheader("🚰 Drinking Quality")
        st.metric("Drinking WQI", f"{wqi:.2f}")
        st.write(f"Class: **{'Excellent' if wqi < 50 else 'Good' if wqi < 100 else 'Unsuitable'}**")
    with res2:
        st.subheader("🌾 Irrigation Suitability")
        st.metric("SAR Index", f"{sar:.2f}")
        st.write(f"Na%: **{df['Na_percent'].iloc[0]:.2f}%**")

with tab2:
    plot_option = st.selectbox("Select Diagram:", 
        ["Gibbs", "Triangle Piper", "Rectangle Piper", "Durov", "HFE-D", "Stiff", "Chadha", "Gaillardet", "Schoeller", "Chernoff"])
    
    plot_map = {
        "Gibbs": gibbs, "Triangle Piper": triangle_piper, "Rectangle Piper": rectangle_piper,
        "Durov": durvo, "HFE-D": hfed, "Stiff": stiff, "Chadha": chadha,
        "Gaillardet": gaillardet, "Schoeller": schoeller, "Chernoff": chernoff
    }

    if st.button("Generate Diagram"):
        with contextlib.redirect_stdout(io.StringIO()):
            plot_func = plot_map[plot_option]
            plot_func.plot(df, unit='mg/L', figname="current_plot", figformat='png')
        st.image("current_plot.png", use_container_width=True)

with tab3:
    st.dataframe(df)
