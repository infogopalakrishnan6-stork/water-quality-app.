import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextlib
import io
from wqchartpy import gibbs, triangle_piper, rectangle_piper, durvo, chadha

# App Config
st.set_page_config(page_title="Water Quality App", layout="wide")
st.title("💧 Water Quality & Suitability Analyzer")

# --- Logic ---
WHO_STD = {"pH":8.5, "Ca":75, "Mg":50, "Na":200, "K":12, "HCO3":300, "Cl":250, "SO4":250, "TDS":500}
WEIGHTS = {"pH":4, "Ca":2, "Mg":2, "Na":3, "K":1, "HCO3":2, "Cl":3, "SO4":3, "TDS":4}

def calc_wqi(row):
    total_wt = sum(WEIGHTS.values())
    rel_wt = {k: v/total_wt for k, v in WEIGHTS.items()}
    return sum((row[p]/WHO_STD[p]*100)*rel_wt[p] for p in WHO_STD)

# --- Sidebar Inputs ---
st.sidebar.header("Input Parameters")
ph = st.sidebar.number_input("pH", value=7.8)
ca = st.sidebar.number_input("Ca", value=32.0)
mg = st.sidebar.number_input("Mg", value=6.0)
na = st.sidebar.number_input("Na", value=28.0)
k = st.sidebar.number_input("K", value=2.8)
hco3 = st.sidebar.number_input("HCO3", value=73.0)
cl = st.sidebar.number_input("Cl", value=43.0)
so4 = st.sidebar.number_input("SO4", value=48.0)
tds = st.sidebar.number_input("TDS", value=233.0)

# Build DataFrame
df = pd.DataFrame([{
    "Sample": "Sample_1", "Label": "User", "Color": "red", "Marker": "o", "Size": 50, "Alpha": 0.7,
    "pH": ph, "Ca": ca, "Mg": mg, "Na": na, "K": k, "HCO3": hco3, "CO3": 0, "Cl": cl, "SO4": so4, "TDS": tds
}])

# Results
wqi_val = calc_wqi(df.iloc[0])
st.metric("Drinking Water WQI", f"{wqi_val:.2f}")

# Plotting with Error Protection
st.subheader("Hydrochemical Plot")
if st.button("Generate Gibbs Plot"):
    with contextlib.redirect_stdout(io.StringIO()):
        gibbs.plot(df, unit='mg/L', figname="output_plot", figformat='png')
    st.image("output_plot.png")
