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
    
    # Drinking WQI
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
    
    # Categorization for Analytics
    df['is_drinking_safe'] = df['WQI_Drinking'] <= 100
    df['is_irrigation_safe'] = (df['SAR'] < 18) & (df['RSC'] < 2.5) & (df['Kelly_Ratio'] < 1)
    return df

def get_suitability_report(row):
    d_safe = row['is_drinking_safe']
    i_safe = row['is_irrigation_safe']
    
    if d_safe and i_safe: return "✅ Highly Suitable: Safe for All Uses", "green"
    elif d_safe: return "⚠️ Suitable for Drinking Only", "blue"
    elif i_safe: return "🚜 Suitable for Irrigation Only", "orange"
    else: return "🚨 Not Suitable: Unsafe for Human or Agricultural use", "red"

# 3. Sidebar
st.sidebar.title("🎛️ Data Management")
input_mode = st.sidebar.radio("Input Method:", ["Manual Entry", "Batch Analysis (Upload CSV)"])

if input_mode == "Manual Entry":
    st.sidebar.subheader("Chemical Parameters (mg/L)")
    ph = st.sidebar.number_input("pH", 7.8)
    tds = st.sidebar.number_input("TDS", 233.0)
    ca, mg, na = st.sidebar.number_input("Ca", 32.0), st.sidebar.number_input("Mg", 6.0), st.sidebar.number_input("Na", 28.0)
    k, hco3, cl, so4 = st.sidebar.number_input("K", 2.8), st.sidebar.number_input("HCO3", 73.0), st.sidebar.number_input("Cl", 43.0), st.sidebar.number_input("SO4", 48.0)
    data = {"Sample": ["User_1"], "pH": [ph], "Ca": [ca], "Mg": [mg], "Na": [na], "K": [k], "HCO3": [hco3], "CO3": [0], "Cl": [cl], "SO4": [so4], "TDS": [tds]}
    df = process_data(pd.DataFrame(data))
    selected_idx = 0
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type="csv")
    if uploaded_file is not None:
        df = process_data(pd.read_csv(uploaded_file))
        st.sidebar.divider()
        st.sidebar.subheader("🔎 Select Sample to View")
        sample_choice = st.sidebar.selectbox("Choose Sample:", df['Sample'].tolist())
        selected_idx = df[df['Sample'] == sample_choice].index[0]
    else:
        st.info("Please upload a CSV file to proceed.")
        st.stop()

# 4. Dashboard Header & Analytics
st.markdown("<h1 style='text-align: center; font-weight: 900;'>🌊 WATER QUALITY SUITABILITY DASHBOARD</h1>", unsafe_allow_html=True)

if input_mode == "Batch Analysis (Upload CSV)":
    total = len(df)
    both = len(df[df['is_drinking_safe'] & df['is_irrigation_safe']])
    drink = len(df[df['is_drinking_safe']])
    irrig = len(df[df['is_irrigation_safe']])
    
    st.subheader("📊 Dataset Overview (Overall Statistics)")
    met1, met2, met3 = st.columns(3)
    met1.metric("Safe for Both", f"{(both/total)*100:.1f}%")
    met2.metric("Safe for Drinking", f"{(drink/total)*100:.1f}%")
    met3.metric("Safe for Irrigation", f"{(irrig/total)*100:.1f}%")
    st.divider()

# 5. Selected Sample Results
current_row = df.iloc[selected_idx]
verdict_text, verdict_color = get_suitability_report(current_row)

st.subheader(f"📍 Analysis Result: {current_row['Sample']}")
st.markdown(f"""<div style="background-color:{verdict_color}; padding:20px; border-radius:10px; color:white; font-size:24px; font-weight:bold; text-align:center;">
            {verdict_text}</div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Quality Results", "📈 Hydrochemical Plots", "📑 Full Dataset"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🚰 Drinking Quality")
        st.metric("Drinking WQI", f"{current_row['WQI_Drinking']:.2f}")
        status = "✅ SAFE" if current_row['is_drinking_safe'] else "🚨 UNSUITABLE"
        st.markdown(f"Status: **{status}**")
        
    with c2:
        st.subheader("🌾 Irrigation Suitability")
        st.metric("SAR Index", f"{current_row['SAR']:.2f}")
        status = "✅ SAFE" if current_row['is_irrigation_safe'] else "🚨 UNSUITABLE"
        st.markdown(f"Status: **{status}**")

with tab2:
    plot_option = st.selectbox("Select Diagram Type:", ["Gibbs", "Triangle Piper", "Rectangle Piper", "Durov", "HFE-D", "Stiff", "Chadha", "Gaillardet", "Schoeller", "Chernoff"])
    if st.button("Generate Plot"):
        with contextlib.redirect_stdout(io.StringIO()):
            plot_map = {"Gibbs": gibbs, "Triangle Piper": triangle_piper, "Rectangle Piper": rectangle_piper, "Durov": durvo, "HFE-D": hfed, "Stiff": stiff, "Chadha": chadha, "Gaillardet": gaillardet, "Schoeller": schoeller, "Chernoff": chernoff}
            # For plotting, ensuring basic visual columns exist
            plot_df = df.copy()
            for col, val in [("Color","red"), ("Marker","o"), ("Size",50), ("Alpha",0.8), ("Label","Site")]:
                if col not in plot_df.columns: plot_df[col] = val
            plot_map[plot_option].plot(plot_df, unit='mg/L', figname="water_plot", figformat='png')
        st.image("water_plot.png", use_container_width=True)

with tab3:
    st.dataframe(df)
