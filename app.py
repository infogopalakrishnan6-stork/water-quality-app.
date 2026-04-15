import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextlib
import io
from wqchartpy import (gibbs, triangle_piper, rectangle_piper, durvo, 
                       hfed, stiff, chadha, gaillardet, schoeller, chernoff)

# 1. Config & BIS Standards
st.set_page_config(page_title="Professional Water Suitability Suite", layout="wide")

# Updated BIS standards and weights to include NO3 and F
BIS_LIMITS = {
    "pH": 8.5, "SO4": 200, "NO3": 45, "F": 1.5, "Cl": 250, 
    "TDS": 500, "Na": 100, "Ca": 75, "Mg": 30, "K": 10, "HCO3": 200
}
BIS_WEIGHTS = {
    "pH": 4, "SO4": 4, "NO3": 5, "F": 5, "Cl": 5, 
    "TDS": 5, "Na": 4, "Ca": 3, "Mg": 3, "K": 2, "HCO3": 1
}
EQ_WT = {"Ca": 20.04, "Mg": 12.15, "Na": 23, "K": 39.1, "HCO3": 61, "CO3": 30}

# 2. Advanced Logic Functions
def process_data(df):
    potential_names = ['Sample', 'sample', 'SAMPLE', 'Sample No', 'Sample No.', 'Sample no', 'SAMP']
    found_name = next((name for name in potential_names if name in df.columns), df.columns[0])
    df = df.rename(columns={found_name: 'Sample_ID'})

    total_wi = sum(BIS_WEIGHTS.values())
    
    # Drinking WQI Calculation
    def calc_wqi(row):
        wqi_sum = 0
        for p, limit in BIS_LIMITS.items():
            if p in row and not pd.isna(row[p]):
                qi = (row[p] / limit) * 100
                Wi = BIS_WEIGHTS[p] / total_wi
                wqi_sum += qi * Wi
        return wqi_sum

    df["WQI_Value"] = df.apply(calc_wqi, axis=1)
    
    # meq/L Conversions
    for ion, ew in EQ_WT.items():
        if ion in df.columns:
            df[f"{ion}_meq"] = df[ion] / ew
            
    # Irrigation Indices
    # Note: SAR and others require Ca, Mg, Na in meq/L
    if all(col in df.columns for col in ["Na_meq", "Ca_meq", "Mg_meq"]):
        df["SAR"] = df["Na_meq"] / np.sqrt((df["Ca_meq"] + df["Mg_meq"]) / 2)
        df["Na_percent"] = (df["Na_meq"] + df.get("K_meq", 0)) / (df["Ca_meq"] + df["Mg_meq"] + df["Na_meq"] + df.get("K_meq", 0)) * 100
        df["RSC"] = (df.get("HCO3_meq", 0) + df.get("CO3_meq", 0)) - (df["Ca_meq"] + df["Mg_meq"])
        df["Kelly_Ratio"] = df["Na_meq"] / (df["Ca_meq"] + df["Mg_meq"])
        df["Mag_Hazard"] = (df["Mg_meq"] / (df["Ca_meq"] + df["Mg_meq"])) * 100
    
    # Internal Binary check for Statistics
    df['is_drinking_safe'] = df['WQI_Value'] <= 50
    df['is_irrigation_safe'] = (df.get('SAR', 0) < 18) & (df.get('RSC', 0) < 1.25) & (df.get('Kelly_Ratio', 0) < 1)
    
    return df

def get_classification_report(row):
    val = row['WQI_Value']
    if val <= 25: return "Excellent Quality", "green", "Safe for Drinking and Irrigation"
    elif val <= 50: return "Good Quality", "blue", "Safe for Drinking and Irrigation"
    elif val <= 75: return "Moderate Quality", "orange", "Irrigation and treatment needed"
    elif val <= 100: return "Poor Quality", "red", "Attention needed for irrigation"
    else: return "Very Poor Quality", "darkred", "Unfit for all uses"

# 3. Sidebar UI
st.sidebar.title("🎛️ Data Management")
input_mode = st.sidebar.radio("Input Method:", ["Manual Entry", "Batch Analysis (Upload CSV)"])

if input_mode == "Manual Entry":
    st.sidebar.subheader("Chemical Parameters (mg/L)")
    params = {}
    # Dynamically generate inputs for all BIS parameters including NO3 and F
    for p in BIS_LIMITS.keys():
        params[p] = st.sidebar.number_input(p, value=float(BIS_LIMITS[p]))
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
        st.info("Please upload a CSV file to begin.")
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
v_text, v_color, v_usage = get_classification_report(current_row)

st.subheader(f"📍 Detailed Analysis: {current_row['Sample_ID']}")
st.markdown(f"""<div style="background-color:{v_color}; padding:20px; border-radius:10px; color:white; font-size:24px; font-weight:bold; text-align:center; margin-bottom:25px;">
            {v_text} | {v_usage}</div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Quality Indices", "📈 Hydrochemical Plots", "📑 Dataset View"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🚰 Drinking Quality")
        st.metric("Drinking WQI", f"{current_row['WQI_Value']:.2f}")
        st.markdown(f"**Overall Status:** {v_text}")
        
    with col2:
        st.subheader("🌾 Irrigation Suitability")
        i_col1, i_col2 = st.columns(2)
        i_col1.metric("SAR Index", f"{current_row.get('SAR', 0):.2f}")
        i_col1.metric("Kelly Ratio", f"{current_row.get('Kelly_Ratio', 0):.2f}")
        i_col2.metric("Na %", f"{current_row.get('Na_percent', 0):.1f}%")
        i_col2.metric("RSC Index", f"{current_row.get('RSC', 0):.2f}")
        st.metric("Magnesium Hazard", f"{current_row.get('Mag_Hazard', 0):.1f}%")
    
    st.divider()
    
    with st.expander("📖 Scientific Interpretation Guide (How to read these results)"):
        st.markdown("""
        ### **1. Drinking Quality (WQI)**
        - **0 - 25**: Excellent Quality. Safe for direct consumption.
        - **26 - 50**: Good Quality. Safe for consumption.
        - **51 - 75**: Moderate Quality. Irrigation and treatment needed before drinking.
        - **76 - 100**: Poor Quality. Need attention for irrigation.
        - **> 100**: Very Poor Quality. Unfit for all uses.
        
        ### **2. Irrigation Indices**
        - **SAR (Sodium Adsorption Ratio)**: Excellent (<10), Good (10-18), Doubtful (18-26), Unfit (>26).
        - **Na % (Percentage Sodium)**: Excellent (<20%), Good (20-40%), Permissible (40-60%), Doubtful (60-80%), Unfit (>80%).
        - **RSC (Residual Sodium Carbonate)**: Good (<1.25), Doubtful (1.25-2.5), Unfit (>2.5).
        - **Kelly Ratio**: Target: **< 1.0**. Values **> 1.0** indicate sodium excess.
        - **Magnesium Hazard**: Suitable (<50%), Unsuitable (>50%).
        """)

with tab2:
    plot_option = st.selectbox("Select Diagram Type:", ["Gibbs", "Triangle Piper", "Rectangle Piper", "Durov", "HFE-D", "Stiff", "Chadha", "Gaillardet", "Schoeller", "Chernoff"])
    if st.button("Generate Selected Plot"):
        with contextlib.redirect_stdout(io.StringIO()):
            plot_df = df.copy().rename(columns={'Sample_ID': 'Sample'})
            for col, val in [("Color","blue"), ("Marker","o"), ("Size",50), ("Alpha",0.8), ("Label","Site")]:
                if col not in plot_df.columns: plot_df[col] = val
            
            plot_map = {"Gibbs": gibbs, "Triangle Piper": triangle_piper, "Rectangle Piper": rectangle_piper, "Durov": durvo, "HFE-D": hfed, "Stiff": stiff, "Chadha": chadha, "Gaillardet": gaillardet, "Schoeller": schoeller, "Chernoff": chernoff}
            plot_map[plot_option].plot(plot_df, unit='mg/L', figname="water_plot", figformat='png')
        st.image("water_plot.png", use_container_width=True)

with tab3:
    st.dataframe(df)
