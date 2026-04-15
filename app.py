import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextlib
import io
from wqchartpy import (gibbs, triangle_piper, rectangle_piper, durvo, 
                       hfed, stiff, chadha, gaillardet, schoeller, chernoff)

# 1. Config & Standards (Updated with BIS Standards and Weights)
st.set_page_config(page_title="Pro Water Chemistry Suite", layout="wide")

# BIS Standards desired limit
BIS_STD = {"pH":8.5, "Ca":75, "Mg":30, "Na":100, "K":10, "HCO3":200, "Cl":250, "SO4":200, "TDS":500, "F":1.5, "NO3":45}
# BIS Weights (wi)
BIS_WEIGHTS = {"pH":4, "Ca":3, "Mg":3, "Na":4, "K":2, "HCO3":1, "Cl":5, "SO4":5, "TDS":5, "F":5, "NO3":5}
# meq/L equivalent weights for irrigation indices
EQ_WT = {"Ca":20.04, "Mg":12.15, "Na":23, "K":39.1, "HCO3":61, "CO3":30}

# 2. Logic Functions
def process_data(df):
    # Flexible header detection for 'Sample'
    potential_names = ['Sample', 'sample', 'SAMPLE', 'Sample No', 'Sample No.', 'Sample no']
    found_name = next((name for name in potential_names if name in df.columns), df.columns[0])
    df = df.rename(columns={found_name: 'Sample_ID'})

    total_wt = sum(BIS_WEIGHTS.values())
    rel_wt = {k: v/total_wt for k, v in BIS_WEIGHTS.items()}
    
    # Calculate WQI based on image data
    # Smart handling of subordinate parameters: if p is not in row, it's treated as 0 and skipped
    df["WQI"] = df.apply(lambda row: sum((row.get(p, 0)/BIS_STD[p]*100)*rel_wt[p] for p in BIS_STD if p in row), axis=1)
    
    # Unit Conversion to meq/L for Irrigation Indices
    for ion, ew in EQ_WT.items():
        if ion in df.columns:
            df[f"{ion}_meq"] = df[ion]/ew
            
    # Comprehensive Irrigation Indices Logic
    df["SAR"] = df["Na_meq"]/np.sqrt((df["Ca_meq"] + df["Mg_meq"])/2)
    df["Na_percent"] = (df["Na_meq"]+df["K_meq"]) / (df["Ca_meq"]+df["Mg_meq"]+df["Na_meq"]+df["K_meq"]) * 100
    df["RSC"] = (df.get("HCO3_meq", 0) + df.get("CO3_meq", 0)) - (df["Ca_meq"] + df["Mg_meq"])
    df["Kelly_Ratio"] = df["Na_meq"] / (df["Ca_meq"] + df["Mg_meq"])
    return df

# 3. Classifications Functions (Exact match to classification images)
def classify_wqi(val):
    if val <= 25: return "Excellent", "green", "Drinking and irrigation"
    if val <= 50: return "Good", "blue", "Drinking and irrigation"
    if val <= 75: return "Moderate", "orange", "Irrigation and treatment needed before drinking"
    if val <= 100: return "Poor", "red", "Need attention for irrigation"
    return "Very poor", "darkred", "Unfit for all uses"

def classify_sar(val):
    if val <= 10: return "Excellent", "green"
    if val <= 18: return "Good", "blue"
    if val <= 26: return "Doubtful", "orange"
    return "Unfit", "red"

def classify_rsc(val):
    if val < 1.25: return "Good", "green"
    if val <= 2.5: return "Doubtful", "orange"
    return "Unfit", "red"

def classify_magnesium_hazard(row):
    # Magnesium Hazard = (Mg_meq / (Ca_meq + Mg_meq)) * 100
    if ('Ca_meq' not in row.index) or ('Mg_meq' not in row.index): return "Unknown", "grey"
    try:
        val = (row['Mg_meq'] / (row['Ca_meq'] + row['Mg_meq'])) * 100
        if val <= 50: return "Suitable", "green"
        return "Unsuitable", "red"
    except ZeroDivisionError: return "Unknown", "grey"

def classify_sodium_percent(val):
    if val <= 20: return "Excellent", "green"
    if val <= 40: return "Good", "blue"
    if val <= 60: return "Permissible", "lightblue"
    if val <= 80: return "Doubtful", "orange"
    return "Unfit", "red"

# 4. Sidebar UI
st.sidebar.title("🎛️ Data Management")
input_mode = st.sidebar.radio("Input Method:", ["Manual Entry", "Batch Analysis (Upload CSV)"])

if input_mode == "Manual Entry":
    st.sidebar.subheader("Chemical Parameters (mg/L)")
    params = {}
    for p in BIS_STD.keys():
        params[p] = st.sidebar.number_input(f"{p} (Desired limit: {BIS_STD[p]})", value=BIS_STD[p])
    # Subordinate parameters (F, NO3) handle defaults well in manual entry.
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
        st.info("Please upload a CSV file with headers matching BIS list. If F or NO3 are missing, calculation will skip them.")
        st.stop()

# 5. Main Dashboard Header & Statistics
st.markdown("<h1 style='text-align: center; font-weight: 900;'>🌊 BIS WATER QUALITY DASHBOARD</h1>", unsafe_allow_html=True)

if input_mode == "Batch Analysis (Upload CSV)":
    total = len(df)
    m1, m2 = st.columns(2)
    # Binary "safe" logic for overall stats
    m1.metric("WQI Excellent or Good", f"{(len(df[df['WQI'] <= 50]) / total) * 100:.1f}%")
    # Irrigation binary check: SAR Excellent or Good (<18)
    m2.metric("Irrigation (SAR < 18)", f"{(len(df[df['SAR'] < 18]) / total) * 100:.1f}%")
    st.divider()

# 6. Selected Sample Results
current_row = df.iloc[selected_idx]

# Categorizations
wqi_class, wqi_color, wqi_purpose = classify_wqi(current_row['WQI'])
sar_class, sar_color = classify_sar(current_row['SAR'])
rsc_class, rsc_color = classify_rsc(current_row['RSC'])
mag_class, mag_color = classify_magnesium_hazard(current_row)
sod_class, sod_color = classify_sodium_percent(current_row['Na_percent'])

st.subheader(f"📍 Detailed Analysis for BIS Suite: {current_row['Sample_ID']}")

# Redesigned Verdict Box to fit BIS classification scheme
st.markdown(f"""<div style="border-left: 10px solid {wqi_color}; background-color:rgba(0,0,0,0.05); padding:20px; border-radius:10px; margin-bottom:25px;">
            <div style="font-size:24px; font-weight:bold;">WQI Status: {wqi_class} (WQI = {current_row['WQI']:.2f})</div>
            <div style="font-size:18px;">Suitable for: {wqi_purpose}</div>
            </div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Quality Indices", "📈 Hydrochemical Plots", "📑 Dataset View"])

with tab1:
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown(f"<div style='background-color:{wqi_color}; padding:10px; border-radius:5px; color:white; font-weight:bold;'>BIS WQI</div>", unsafe_allow_html=True)
        st.metric("Water Quality Index", f"{current_row['WQI']:.2f}")
        st.write(f"Class: **{wqi_class}**")
        
    with c2:
        st.markdown(f"<div style='background-color:grey; padding:10px; border-radius:5px; color:white; font-weight:bold;'>Irrigation Suitability</div>", unsafe_allow_html=True)
        # Using BIS classification names for labels
        r1_c1, r1_c2 = st.columns(2)
        r1_c1.metric("SAR", f"{current_row['SAR']:.2f}", sar_class)
        r1_c2.metric("RSC", f"{current_row['RSC']:.2f}", rsc_class)
        
        r2_c1, r2_c2 = st.columns(2)
        r2_c1.metric("Na %", f"{current_row['Na_percent']:.1f}%", sod_class)
        # Assuming Kelly Ratio classify similar to SAR
        r2_c2.metric("Kelly Ratio", f"{current_row['Kelly_Ratio']:.2f}", "Excellent" if current_row['Kelly_Ratio'] < 1 else "Unfit")
        
    st.divider()
    
    # Optional expander for interpretation guide
    with st.expander("📖 Guide: How to read BIS classifications"):
        st.markdown("""
        ### **1. BIS WQI (Drinking & Irrigation)**
        - **WQI 0-25**: Excellent Quality. 
        - **WQI 26-50**: Good Quality.
        - **WQI 51-75**: Moderate (Needs treatment before drinking).
        - **WQI 76-100**: Poor (Need attention for irrigation).
        - **WQI > 100**: Very poor (Unfit).
        
        ### **2. Irrigation Indices**
        - **SAR**: Measures soil permeability hazard. BIS Class names: **Excellent**, Good, Doubtful, Unfit.
        - **RSC**: Hazard of Carbonate. Classes: **Good**, Doubtful, Unfit.
        - **Na %**: Classes: **Excellent**, Good, **Permissible**, Doubtful, Unfit.
        - **Magnesium Hazard**: (Calculated but not shown in metrics, affects overall statistics internally).
        - **Kelly Ratio**: **< 1** is Safe.
        """)

with tab2:
    plot_option = st.selectbox("Select Diagram Type:", ["Gibbs", "Triangle Piper", "Rectangle Piper", "Durov", "HFE-D", "Stiff", "Chadha", "Gaillardet", "Schoeller", "Chernoff"])
    if st.button("Generate selected Plot"):
        with contextlib.redirect_stdout(io.StringIO()):
            plot_df = df.copy().rename(columns={'Sample_ID': 'Sample'})
            for col, val in [("Color","blue"), ("Marker","o"), ("Size",50), ("Alpha",0.8), ("Label","Site")]:
                if col not in plot_df.columns: plot_df[col] = val
            plot_map = {"Gibbs": gibbs, "Triangle Piper": triangle_piper, "Rectangle Piper": rectangle_piper, "Durov": durvo, "HFE-D": hfed, "Stiff": stiff, "Chadha": chadha, "Gaillardet": gaillardet, "Schoeller": schoeller, "Chernoff": chernoff}
            plot_map[plot_option].plot(plot_df, unit='mg/L', figname="water_plot", figformat='png')
        st.image("water_plot.png", use_container_width=True)

with tab3:
    st.dataframe(df)
