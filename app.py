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

# 2. Logic Functions
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

def get_suitability_report(wqi, sar, rsc, kelly):
    # Logic for Verdict
    is_drinkable = wqi <= 100
    is_irrigation = sar < 18 and rsc < 2.5 and kelly < 1
    
    if is_drinkable and is_irrigation:
        return "✅ Highly Suitable: Safe for Drinking & Irrigation", "green"
    elif is_drinkable:
        return "⚠️ Suitable for Drinking Only: High Mineral Hazard for Soil", "blue"
    elif is_irrigation:
        return "🚜 Suitable for Irrigation Only: Exceeds Drinking Safety Limits", "orange"
    else:
        return "🚨 Not Suitable: Unsafe for Human or Agricultural use", "red"

# 3. Sidebar
st.sidebar.header("Input Chemical Values")
ph = st.sidebar.number_input("pH", 7.8)
tds = st.sidebar.number_input("TDS (mg/L)", 233.0)
ca = st.sidebar.number_input("Ca (mg/L)", 32.0)
mg = st.sidebar.number_input("Mg (mg/L)", 6.0)
na = st.sidebar.number_input("Na (mg/L)", 28.0)
k = st.sidebar.number_input("K (mg/L)", 2.8)
hco3 = st.sidebar.number_input("HCO3 (mg/L)", 73.0)
cl = st.sidebar.number_input("Cl (mg/L)", 43.0)
so4 = st.sidebar.number_input("SO4 (mg/L)", 48.0)

# 4. Data Processing
data = {
    "Sample": ["User"], "Label": ["Manual"], "Color": ["blue"], "Marker": ["o"], "Size": [50], "Alpha": [0.8],
    "pH": [ph], "Ca": [ca], "Mg": [mg], "Na": [na], "K": [k], "HCO3": [hco3], "CO3": [0], "Cl": [cl], "SO4": [so4], "TDS": [tds]
}
df = process_data(pd.DataFrame(data))
wqi, sar, rsc, kelly = df["WQI_Drinking"].iloc[0], df["SAR"].iloc[0], df["RSC"].iloc[0], df["Kelly_Ratio"].iloc[0]

# 5. UI Layout
st.title("💧 Water Quality & Suitability Dashboard")

# Top Verdict Box
verdict, color = get_suitability_report(wqi, sar, rsc, kelly)
st.markdown(f"""<div style="background-color:{color}; padding:20px; border-radius:10px; color:white; font-size:24px; font-weight:bold; text-align:center;">
            {verdict}</div>""", unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Drinking Assessment")
    st.metric("Drinking WQI", f"{wqi:.2f}")
    st.progress(min(wqi/300, 1.0))
    st.write(f"Status: **{'Safe' if wqi <= 100 else 'Unsafe'}**")

with col2:
    st.subheader("🌱 Irrigation Assessment")
    st.metric("SAR Index", f"{sar:.2f}")
    st.write(f"RSC Value: **{rsc:.2f}**")
    st.write(f"Kelly Ratio: **{kelly:.2f}**")

st.divider()

# Information Guide
with st.expander("📖 Help: How to interpret these results?"):
    st.write("""
    * **WQI**: Below 100 is considered fit for human consumption.
    * **SAR**: Values below 10 are excellent for irrigation. High SAR causes soil to lose its permeability.
    * **RSC**: Values below 1.25 are safe. Over 2.5 is unsuitable as it indicates a sodium hazard.
    """)

# Plot Section
st.subheader("📈 Hydrochemical Plots")
if st.button("Generate Gibbs Plot"):
    with contextlib.redirect_stdout(io.StringIO()):
        gibbs.plot(df, unit='mg/L', figname="gibbs_plot", figformat='png')
    st.image("gibbs_plot.png")
