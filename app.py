import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextlib
import io
from wqchartpy import (gibbs, triangle_piper, rectangle_piper, durvo, 
                       hfed, stiff, chadha, gaillardet, schoeller, chernoff)

# 1. BIS Standards & Weightage (Source: User Images)
st.set_page_config(page_title="BIS Water Quality Suite", layout="wide")

# BIS standards desired limits and weights (wi)
BIS_LIMITS = {
    "pH": 8.5, "SO4": 200, "NO3": 45, "F": 1.5, "Cl": 250, 
    "TDS": 500, "Na": 100, "Ca": 75, "Mg": 30, "K": 10, "HCO3": 200
}
BIS_WEIGHTS = {
    "pH": 4, "SO4": 5, "NO3": 5, "F": 5, "Cl": 5, 
    "TDS": 5, "Na": 4, "Ca": 3, "Mg": 3, "K": 2, "HCO3": 1
}
EQ_WT = {"Ca": 20.04, "Mg": 12.15, "Na": 23, "K": 39.1, "HCO3": 61, "CO3": 30}

# 2. Logic & Classification Functions
def process_data(df):
    # Detect Sample Column
    potential_names = ['Sample', 'sample', 'SAMPLE', 'Sample No', 'Sample No.', 'Sample no']
    found_name = next((name for name in potential_names if name in df.columns), df.columns[0])
    df = df.rename(columns={found_name: 'Sample_ID'})

    # Calculate Relative Weight (Wi)
    total_wi = sum(BIS_WEIGHTS.values())
    
    # Calculate WQI
    def calc_wqi(row):
        wqi_sum = 0
        for p, limit in BIS_LIMITS.items():
            if p in row and not pd.isna(row[p]):
                # Quality Rating (qi) = (Ci/Si) * 100
                qi = (row[p] / limit) * 100
                # Wi = wi / sum(wi)
                Wi = BIS_WEIGHTS[p] / total_wi
                wqi_sum += qi * Wi
        return wqi_sum

    df["WQI"] = df.apply(calc_wqi, axis=1)
    
    # meq/L Conversions for Irrigation
    for ion, ew in EQ_WT.items():
        if ion in df.columns:
            df[f"{ion}_meq"] = df[ion] / ew
            
    # Irrigation Indices
    df["SAR"] = df["Na_meq"] / np.sqrt((df["Ca_meq"] + df["Mg_meq"]) / 2)
    df["RSC"] = (df.get("HCO3_meq", 0) + df.get("CO3_meq", 0)) - (df["Ca_meq"] + df["Mg_meq"])
    df["Na_percent"] = ((df["Na_meq"] + df["K_meq"]) / (df["Ca_meq"] + df["Mg_meq"] + df["Na_meq"] + df["K_meq"])) * 100
    df["Kelly_Ratio"] = df["Na_meq"] / (df["Ca_meq"] + df["Mg_meq"])
    df["Magnesium_Hazard"] = (df["Mg_meq"] / (df["Ca_meq"] + df["Mg_meq"])) * 100
    
    return df

# Classification Logic based on provided charts
def get_wqi_class(val):
    if val <= 25: return "Excellent", "green", "Drinking and irrigation"
    elif val <= 50: return "Good", "blue", "Drinking and irrigation"
    elif val <= 75: return "Moderate", "orange", "Irrigation and treatment needed"
    elif val <= 100: return "Poor", "red", "Attention needed for irrigation"
    else: return "Very Poor", "darkred", "Unfit for all uses"

def get_sar_class(val):
    if val <= 10: return "Excellent", "green"
    elif val <= 18: return "Good", "blue"
    elif val <= 26: return "Doubtful", "orange"
    else: return "Unfit", "red"

def get_rsc_class(val):
    if val < 1.25: return "Good", "green"
    elif val <= 2.5: return "Doubtful", "orange"
    else: return "Unfit", "red"

def get_na_class(val):
    if val <= 20: return "Excellent", "green"
    elif val <= 40: return "Good", "blue"
    elif val <= 60: return "Permissible", "lightblue"
    elif val <= 80: return "Doubtful", "orange"
    else: return "Unfit", "red"

# 3. Sidebar
st.sidebar.title("🎛️ Data Management")
input_mode = st.sidebar.radio("Input Method:", ["Manual Entry", "Batch Analysis (Upload CSV)"])

if input_mode == "Manual Entry":
    st.sidebar.subheader("Chemical Parameters (mg/L)")
    params = {}
    for p in BIS_LIMITS.keys():
        params[p] = st.sidebar.number_input(f"{p} (Limit: {BIS_LIMITS[p]})", value=float(BIS_LIMITS[p]))
    data = {"Sample_ID": ["Manual_1"], **{k: [v] for k, v in params.items()}, "CO3": [0]}
    df = process_data(pd.DataFrame(data))
    selected_idx = 0
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type="csv")
    if uploaded_file is not None:
        df = process_data(pd.read_csv(uploaded_file))
        st.sidebar.divider()
        sample_choice = st.sidebar.selectbox("🔎 Select Sample:", df['Sample_ID'].tolist())
        selected_idx = df[df['Sample_ID'] == sample_choice].index[0]
    else:
        st.info("Please upload a CSV to begin.")
        st.stop()

# 4. Main Dashboard
st.markdown("<h1 style='text-align: center; font-weight: 900;'>🌊 BIS WATER QUALITY DASHBOARD</h1>", unsafe_allow_html=True)

if input_mode == "Batch Analysis (Upload CSV)":
    total = len(df)
    m1, m2, m3 = st.columns(3)
    m1.metric("Excellent/Good WQI", f"{(len(df[df['WQI'] <= 50]) / total) * 100:.1f}%")
    m2.metric("Safe SAR (<18)", f"{(len(df[df['SAR'] <= 18]) / total) * 100:.1f}%")
    m3.metric("Safe RSC (<1.25)", f"{(len(df[df['RSC'] < 1.25]) / total) * 100:.1f}%")
    st.divider()

# 5. Selected Result
current_row = df.iloc[selected_idx]
w_class, w_color, w_usage = get_wqi_class(current_row['WQI'])

st.subheader(f"📍 Sample Analysis: {current_row['Sample_ID']}")
st.markdown(f"""<div style="background-color:{w_color}; padding:15px; border-radius:10px; color:white; font-size:22px; font-weight:bold; text-align:center;">
            Class: {w_class} | Suitable for: {w_usage}</div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Hydrochemical Plots", "📑 Data Table"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🚰 Drinking Analysis")
        st.metric("Final WQI", f"{current_row['WQI']:.2f}")
        st.write(f"Status: **{w_class}**")
    with c2:
        st.subheader("🌾 Irrigation Indices")
        i_c1, i_c2 = st.columns(2)
        
        sar_label, _ = get_sar_class(current_row['SAR'])
        i_c1.metric("SAR", f"{current_row['SAR']:.2f}", sar_label)
        
        na_label, _ = get_na_class(current_row['Na_percent'])
        i_c1.metric("Na %", f"{current_row['Na_percent']:.1f}%", na_label)
        
        rsc_label, _ = get_rsc_class(current_row['RSC'])
        i_c2.metric("RSC", f"{current_row['RSC']:.2f}", rsc_label)
        
        mag_label = "Suitable" if current_row['Magnesium_Hazard'] <= 50 else "Unsuitable"
        i_c2.metric("Mg Hazard", f"{current_row['Magnesium_Hazard']:.1f}%", mag_label)

with tab2:
    plot_option = st.selectbox("Select Plot:", ["Gibbs", "Triangle Piper", "Durov", "Chadha", "Schoeller"])
    if st.button("Generate Plot"):
        with contextlib.redirect_stdout(io.StringIO()):
            plot_df = df.copy().rename(columns={'Sample_ID': 'Sample'})
            for col, val in [("Color","blue"), ("Marker","o"), ("Size",50), ("Alpha",0.8)]:
                if col not in plot_df.columns: plot_df[col] = val
            
            plot_map = {"Gibbs": gibbs, "Triangle Piper": triangle_piper, "Durov": durvo, "Chadha": chadha, "Schoeller": schoeller}
            plot_map[plot_option].plot(plot_df, unit='mg/L', figname="water_plot", figformat='png')
        st.image("water_plot.png")

with tab3:
    st.dataframe(df)
