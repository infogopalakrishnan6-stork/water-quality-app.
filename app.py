import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextlib
import io
from wqchartpy import gibbs

# 1. Config & Standards
st.set_page_config(page_title="Water Suitability App", layout="wide")

WHO_STD = {"pH":8.5, "Ca":75, "Mg":50, "Na":200, "K":12, "HCO3":300, "Cl":250, "SO4":250, "TDS":500}
WEIGHTS = {"pH":4, "Ca":2, "Mg":2, "Na":3, "K":1, "HCO3":2, "Cl":3, "SO4":3, "TDS":4}
EQ_WT = {"Ca":20.04, "Mg":12.15, "Na":23, "K":39.1, "HCO3":61, "CO3":30}

# 2. Processing Functions
def process_data(df):
    total_wt = sum(WEIGHTS.values())
    rel_wt = {k: v/total_wt for k, v in WEIGHTS.items()}
    df["WQI_Drinking"] = df.apply(lambda row: sum((row[p]/WHO_STD[p]*100)*rel_wt[p] for p in WHO_STD), axis=1)
    
    for ion, ew in EQ_WT.items():
        df[f"{ion}_meq"] = df[ion]/ew
        
    df["SAR"] = df["Na_meq"]/np.sqrt((df["Ca_meq"] + df["Mg_meq"])/2)
    df["Na_percent"] = (df["Na_meq"]+df["K_meq"]) / (df["Ca_meq"]+df["Mg_meq"]+df["Na_meq"]+df["K_meq"]) * 100
    df["RSC"] = (df["HCO3_meq"]+df["CO3_meq"]) - (df["Ca_meq"]+df["Mg_meq"])
    df["Kelly_Ratio"] = df["Na_meq"] / (df["Ca_meq"] + df["Mg_meq"])
    return df

def classify_wqi(wqi):
    if wqi < 50: return "Excellent"
    elif wqi < 100: return "Good"
    elif wqi < 200: return "Poor"
    elif wqi < 300: return "Very Poor"
    else: return "Unsuitable"

def classify_sar(sar):
    if sar < 10: return "Excellent"
    elif sar < 18: return "Good"
    elif sar < 26: return "Doubtful"
    else: return "Unsuitable"

# 3. UI Section
st.title("💧 Water Quality Suitability Dashboard")

st.sidebar.header("Input Parameters")
ph = st.sidebar.number_input("pH", 7.0)
tds = st.sidebar.number_input("TDS (mg/L)", 200.0)
ca = st.sidebar.number_input("Ca (mg/L)", 30.0)
mg = st.sidebar.number_input("Mg (mg/L)", 10.0)
na = st.sidebar.number_input("Na (mg/L)", 20.0)
k = st.sidebar.number_input("K (mg/L)", 2.0)
hco3 = st.sidebar.number_input("HCO3 (mg/L)", 100.0)
co3 = st.sidebar.number_input("CO3 (mg/L)", 0.0)
cl = st.sidebar.number_input("Cl (mg/L)", 30.0)
so4 = st.sidebar.number_input("SO4 (mg/L)", 40.0)

# 4. Calculation
data = {
    "Sample": ["User"], "Label": ["Manual"], "Color": ["blue"], "Marker": ["o"], "Size": [50], "Alpha": [0.8],
    "pH": [ph], "Ca": [ca], "Mg": [mg], "Na": [na], "K": [k], "HCO3": [hco3], "CO3": [co3], "Cl": [cl], "SO4": [so4], "TDS": [tds]
}
df = pd.DataFrame(data)
df = process_data(df)

# 5. Display Dashboard
col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Drinking Water Quality")
    wqi = df["WQI_Drinking"].iloc[0]
    st.metric("WQI", f"{wqi:.2f}")
    st.success(f"Class: {classify_wqi(wqi)}")

with col2:
    st.subheader("🌱 Irrigation Suitability")
    sar = df["SAR"].iloc[0]
    st.metric("SAR", f"{sar:.2f}")
    st.info(f"Class: {classify_sar(sar)}")

st.subheader("🧪 Hydrochemical Parameters (meq/L)")
meq_cols = [col for col in df.columns if "_meq" in col]
st.dataframe(df[meq_cols].round(3))

st.subheader("📌 Additional Indices")
c1, c2, c3 = st.columns(3)
c1.write(f"**Na%:** {df['Na_percent'].iloc[0]:.2f}")
c2.write(f"**RSC:** {df['RSC'].iloc[0]:.2f}")
c3.write(f"**Kelly Ratio:** {df['Kelly_Ratio'].iloc[0]:.2f}")

st.subheader("📈 Hydrochemical Plots")
if st.button("Generate Gibbs Plot"):
    with contextlib.redirect_stdout(io.StringIO()):
        gibbs.plot(df, unit='mg/L', figname="gibbs_plot", figformat='png')
    st.image("gibbs_plot.png")
