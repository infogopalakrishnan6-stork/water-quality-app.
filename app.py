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

# 2. Advanced Logic Functions
def process_data(df):
    # Flexible header detection for 'Sample'
    potential_names = ['Sample', 'sample', 'SAMPLE', 'Sample No', 'Sample No.', 'Sample no']
    found_name = next((name for name in potential_names if name in df.columns), df.columns[0])
    df = df.rename(columns={found_name: 'Sample_ID'})

    total_wt = sum(WEIGHTS.values())
    rel_wt = {k: v/total_wt for k, v in WEIGHTS.items()}
    
    # Drinking WQI
    df["WQI_Drinking"] = df.apply(lambda row: sum((row.get(p, 0)/WHO_STD[p]*100)*rel_wt[p] for p in WHO_STD if p in row), axis=1)
    
    # meq/L Conversions
    for ion, ew in EQ_WT.items():
        if ion in df.columns:
            df[f"{ion}_meq"] = df[ion]/ew
            
    # Comprehensive Irrigation Indices
    df["SAR"] = df["Na_meq"]/np.sqrt((df["Ca_meq"] + df["Mg_meq"])/2)
    df["Na_percent"] = (df["Na_meq"]+df["K_meq"]) / (df["Ca_meq"]+df["Mg_meq"]+df["Na_meq"]+df["K_meq"]) * 100
    df["RSC"] = (df.get("HCO3_meq", 0) + df.get("CO3_meq", 0)) - (df["Ca_meq"] + df["Mg_meq"])
    df["Kelly_Ratio"] = df["Na_meq"] / (df["Ca_meq"] + df["Mg_meq"])
    
    # Binary Status for Statistics
    df['is_drinking_safe'] = df['WQI_Drinking'] <= 100
    # Safe irrigation criteria: SAR < 18, RSC < 1.25, Kelly < 1
    df['is_irrigation_safe'] = (df['SAR'] < 18) & (df['RSC'] < 1.25) & (df['Kelly_Ratio'] < 1)
    return df

def get_suitability_report(row):
    d_safe, i_safe = row['is_drinking_safe'], row['is_irrigation_safe']
    if d_safe and i_safe: return "✅ Highly Suitable: Safe for All Uses", "green"
    elif d_safe: return "⚠️ Suitable for Drinking Only", "blue"
    elif i_safe: return "🚜 Suitable for Irrigation Only", "orange"
    else: return "🚨 Not Suitable: Unsafe for Human or Agricultural use", "red"

# 3. Sidebar UI
st.sidebar.title("🎛️ Data Management")
input_mode = st.sidebar.radio("Input Method:", ["Manual Entry", "Batch Analysis (Upload CSV)"])

if input_mode == "Manual Entry":
    st.sidebar.subheader("Chemical Parameters (mg/L)")
    params = {}
    for p in ["pH", "TDS", "Ca", "Mg", "Na", "K", "HCO3", "Cl", "SO4"]:
        params[p] = st.sidebar.number_input(p, value=7.0 if p == "pH" else 10.0)
    data = {"Sample_ID": ["Manual_Sample"], **{k: [v] for k, v in params.items()}, "CO3": [0]}
    df = process_data(pd.DataFrame(data))
    selected_idx = 0
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type="csv")
    if uploaded_file is not None:
        df = process_data(pd.read_csv(uploaded_file))
        st.sidebar.divider()
        sample_choice = st.sidebar.selectbox("🔎 Select Sample to View:", df['Sample_ID'].tolist())
        selected_idx = df[df['Sample_ID'] == sample_choice].index[0]
    else:
        st.info("Please upload a CSV file to view statistics and specific results.")
        st.stop()

# 4. Main Dashboard Header
st.markdown("<h1 style='text-align: center; font-weight: 900;'>🌊 WATER QUALITY SUITABILITY DASHBOARD</h1>", unsafe_allow_html=True)

if input_mode == "Batch Analysis (Upload CSV)":
    total = len(df)
    m1, m2, m3 = st.columns(3)
    m1.metric("Safe for Both", f"{(len(df[df['is_drinking_safe'] & df['is_irrigation_safe']]) / total) * 100:.1f}%")
    m2.metric("Safe for Drinking", f"{(len(df[df['is_drinking_safe']]) / total) * 100:.1f}%")
    m3.metric("Safe for Irrigation", f"{(len(df[df['is_irrigation_safe']]) / total) * 100:.1f}%")
    st.divider()

# 5. Result Display
current_row = df.iloc[selected_idx]
verdict_text, verdict_color = get_suitability_report(current_row)

st.subheader(f"📍 Detailed Analysis: {current_row['Sample_ID']}")
st.markdown(f"""<div style="background-color:{verdict_color}; padding:20px; border-radius:10px; color:white; font-size:24px; font-weight:bold; text-align:center; margin-bottom:25px;">
            {verdict_text}</div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Quality Indices", "📈 Hydrochemical Plots", "📑 Dataset View"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚰 Drinking Quality")
        st.metric("Drinking WQI", f"{current_row['WQI_Drinking']:.2f}")
        status = "✅ SAFE" if current_row['is_drinking_safe'] else "🚨 UNSUITABLE"
        st.markdown(f"**Overall Status:** {status}")
        
    with col2:
        st.subheader("🌾 Irrigation Suitability")
        i_col1, i_col2 = st.columns(2)
        i_col1.metric("SAR Index", f"{current_row['SAR']:.2f}")
        i_col1.metric("Kelly Ratio", f"{current_row['Kelly_Ratio']:.2f}")
        i_col2.metric("Na %", f"{current_row['Na_percent']:.1f}%")
        i_col2.metric("RSC Index", f"{current_row['RSC']:.2f}")
    
    st.divider()
    
    # 📖 User Guide for Interpretation
    with st.expander("📖 Scientific Interpretation Guide (How to read these results)"):
        st.markdown("""
        ### **1. Drinking Quality (WQI)**
        - **< 50**: Excellent Quality. Safe for direct consumption.
        - **50 - 100**: Good Quality. Safe for consumption.
        - **> 100**: Unsuitable for Drinking. High mineral content or imbalance.
        
        ### **2. Irrigation Indices**
        - **SAR (Sodium Adsorption Ratio)**: Measures the sodium hazard to soil. 
            - *Target:* **< 10** (Excellent). Values **> 26** cause severe soil permeability issues.
        - **Na % (Sodium Percentage)**: Indicates the proportion of sodium. 
            - *Target:* **< 60%**. High sodium percentages stunt plant growth.
        - **RSC (Residual Sodium Carbonate)**: Measures the hazard of bicarbonate/carbonate.
            - *Target:* **< 1.25** (Safe). **> 2.5** is hazardous to soil and crops.
        - **Kelly Ratio**: Ratio of Sodium to Calcium/Magnesium.
            - *Target:* **< 1.0**. Values **> 1.0** indicate sodium excess, making water unsuitable for irrigation.
        """)

with tab2:
    plot_option = st.selectbox("Select Diagram Type:", ["Gibbs", "Triangle Piper", "Rectangle Piper", "Durov", "HFE-D", "Stiff", "Chadha", "Gaillardet", "Schoeller", "Chernoff"])
    if st.button("Generate Selected Plot"):
        with contextlib.redirect_stdout(io.StringIO()):
            # Clean data for plotting
            plot_df = df.copy().rename(columns={'Sample_ID': 'Sample'})
            for col, val in [("Color","blue"), ("Marker","o"), ("Size",50), ("Alpha",0.8), ("Label","Site")]:
                if col not in plot_df.columns: plot_df[col] = val
            
            plot_map = {"Gibbs": gibbs, "Triangle Piper": triangle_piper, "Rectangle Piper": rectangle_piper, "Durov": durvo, "HFE-D": hfed, "Stiff": stiff, "Chadha": chadha, "Gaillardet": gaillardet, "Schoeller": schoeller, "Chernoff": chernoff}
            plot_map[plot_option].plot(plot_df, unit='mg/L', figname="water_plot", figformat='png')
        st.image("water_plot.png", use_container_width=True)

with tab3:
    st.dataframe(df)
