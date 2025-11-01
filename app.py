
# app.py
# NexGen Logistics - Predictive Delivery Optimizer (Streamlit)
# Author: Your Name
# Run: streamlit run app.py
# Python 3.9+

import os
import io
import sys
import json
import time
import math
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

import plotly.express as px

st.set_page_config(page_title="NexGen Predictive Delivery Optimizer", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer):
    try:
        df = pd.read_csv(path_or_buffer)
        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    except Exception as e:
        st.warning(f"Could not read: {path_or_buffer} -> {e}")
        return pd.DataFrame()

def try_parse_datetime(series):
    try:
        return pd.to_datetime(series, errors="coerce")
    except Exception:
        return pd.to_datetime(pd.Series([], dtype="datetime64[ns]"))

def add_kpis_area(kpi_dict):
    cols = st.columns(len(kpi_dict))
    for (label, value), c in zip(kpi_dict.items(), cols):
        c.metric(label, value)

def co2_estimate(distance_km, co2_g_per_km=None, fallback_g_per_km=180):
    # If vehicle data missing, fallback to 180 g/km (light commercial)
    g_per_km = co2_g_per_km if pd.notnull(co2_g_per_km) else fallback_g_per_km
    return (distance_km or 0) * (g_per_km or 0) / 1000.0  # kg CO2

def safe_ratio(n, d):
    return float(n) / float(d) if d else 0.0

# -----------------------------
# Sidebar: Data Sources
# -----------------------------
st.sidebar.title("📦 Data Inputs")
st.sidebar.write("Upload CSVs **or** place them in the `data/` folder with the following names:")
st.sidebar.code("\\n".join([
    "orders.csv",
    "delivery_performance.csv",
    "routes_distance.csv",
    "vehicle_fleet.csv",
    "warehouse_inventory.csv",
    "customer_feedback.csv",
    "cost_breakdown.csv"
]))

uploaded = {}
for name in ["orders.csv","delivery_performance.csv","routes_distance.csv","vehicle_fleet.csv",
             "warehouse_inventory.csv","customer_feedback.csv","cost_breakdown.csv"]:
    uploaded[name] = st.sidebar.file_uploader(f"Upload {name}", type="csv")

# Load with priority: uploaded -> data folder -> empty
dfs = {}
for name, file in uploaded.items():
    if file is not None:
        dfs[name] = load_csv(file)
    else:
        data_path = os.path.join("data", name)
        if os.path.exists(data_path):
            dfs[name] = load_csv(data_path)
        else:
            dfs[name] = pd.DataFrame()

orders = dfs["orders.csv"]
deliv  = dfs["delivery_performance.csv"]
routes = dfs["routes_distance.csv"]
fleet  = dfs["vehicle_fleet.csv"]
wh     = dfs["warehouse_inventory.csv"]
cxf    = dfs["customer_feedback.csv"]
costs  = dfs["cost_breakdown.csv"]

# -----------------------------
# Basic Cleaning & Merging
# -----------------------------
# Expected columns (best-effort)
# orders: order_id, order_date, origin, destination, customer_segment, priority, product_category, order_value
# delivery_performance: order_id, promised_datetime, actual_datetime, carrier, status, rating, delivery_cost, vehicle_id
# routes_distance: order_id, distance_km, fuel_liters, tolls, traffic_delay_min, weather_impact
# vehicle_fleet: vehicle_id, vehicle_type, capacity, fuel_eff_kmpl, co2_g_per_km, age_years
# We will merge on order_id; vehicle join via vehicle_id if available

for df in [orders, deliv, routes, costs]:
    if "order_id" in df.columns:
        # Ensure order_id is str for safe join
        df["order_id"] = df["order_id"].astype(str)

if "vehicle_id" in deliv.columns:
    deliv["vehicle_id"] = deliv["vehicle_id"].astype(str)

# Parse datetimes
if "order_date" in orders.columns:
    orders["order_date"] = try_parse_datetime(orders["order_date"])
if "promised_datetime" in deliv.columns:
    deliv["promised_datetime"] = try_parse_datetime(deliv["promised_datetime"])
if "actual_datetime" in deliv.columns:
    deliv["actual_datetime"]  = try_parse_datetime(deliv["actual_datetime"])

# Derived: delay_minutes, delayed flag
if not deliv.empty and {"promised_datetime","actual_datetime"}.issubset(deliv.columns):
    deliv["delay_minutes"] = (deliv["actual_datetime"] - deliv["promised_datetime"]).dt.total_seconds() / 60.0
    deliv["delayed"] = deliv["delay_minutes"] > 0
else:
    deliv["delay_minutes"] = np.nan
    deliv["delayed"] = np.nan

# Merge data (left joins on orders)
merged = orders.copy()
if not deliv.empty:
    merged = merged.merge(deliv, on="order_id", how="left", suffixes=("","_deliv"))
if not routes.empty:
    merged = merged.merge(routes, on="order_id", how="left")
if not costs.empty:
    merged = merged.merge(costs, on="order_id", how="left", suffixes=("","_cost"))

# Vehicle join (if vehicle_id available)
if "vehicle_id" in merged.columns and not fleet.empty and "vehicle_id" in fleet.columns:
    fleet_cols = [c for c in fleet.columns if c in ["vehicle_id","vehicle_type","capacity","fuel_eff_kmpl","co2_g_per_km","age_years","current_location"]]
    merged = merged.merge(fleet[fleet_cols], on="vehicle_id", how="left")

# CO2 estimate
if "distance_km" in merged.columns:
    merged["co2_kg_est"] = merged.apply(lambda r: co2_estimate(r.get("distance_km", 0), r.get("co2_g_per_km", np.nan)), axis=1)
else:
    merged["co2_kg_est"] = np.nan

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.title("🔎 Filters")
date_min = merged["order_date"].min() if "order_date" in merged.columns and not merged["order_date"].isna().all() else None
date_max = merged["order_date"].max() if "order_date" in merged.columns and not merged["order_date"].isna().all() else None

if date_min and date_max:
    date_range = st.sidebar.date_input("Order Date Range", value=(date_min.date(), date_max.date()))
else:
    date_range = None

priority_opts = sorted([p for p in merged.get("priority", pd.Series(dtype=str)).dropna().unique()]) if "priority" in merged.columns else []
priority_sel = st.sidebar.multiselect("Priority", priority_opts, default=priority_opts[:])

category_opts = sorted([p for p in merged.get("product_category", pd.Series(dtype=str)).dropna().unique()]) if "product_category" in merged.columns else []
category_sel = st.sidebar.multiselect("Product Category", category_opts, default=category_opts[:])

carrier_opts = sorted([p for p in merged.get("carrier", pd.Series(dtype=str)).dropna().unique()]) if "carrier" in merged.columns else []
carrier_sel = st.sidebar.multiselect("Carrier", carrier_opts, default=carrier_opts[:])

warehouse_opts = sorted([p for p in merged.get("origin", pd.Series(dtype=str)).dropna().unique()]) if "origin" in merged.columns else []
warehouse_sel = st.sidebar.multiselect("Origin (Warehouse)", warehouse_opts, default=warehouse_opts[:])

def apply_filters(df):
    out = df.copy()
    if date_range and "order_date" in out.columns:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
        out = out[(out["order_date"] >= start) & (out["order_date"] < end)]
    if priority_sel and "priority" in out.columns:
        out = out[out["priority"].isin(priority_sel)]
    if category_sel and "product_category" in out.columns:
        out = out[out["product_category"].isin(category_sel)]
    if carrier_sel and "carrier" in out.columns:
        out = out[out["carrier"].isin(carrier_sel)]
    if warehouse_sel and "origin" in out.columns:
        out = out[out["origin"].isin(warehouse_sel)]
    return out

filt = apply_filters(merged)

# -----------------------------
# Header & KPIs
# -----------------------------
st.title("🚚 NexGen Predictive Delivery Optimizer")
st.caption("Transforming logistics from reactive to predictive with data-driven insights.")

total_orders = len(filt) if not filt.empty else 0
delayed = int(filt["delayed"].sum()) if "delayed" in filt.columns and filt["delayed"].notna().any() else 0
on_time_rate = f"{100*(1 - safe_ratio(delayed, total_orders)):.1f}%" if total_orders else "N/A"
avg_delay = f"{filt['delay_minutes'].mean():.1f} min" if "delay_minutes" in filt.columns and filt["delay_minutes"].notna().any() else "N/A"
co2_total = f"{filt['co2_kg_est'].sum():.1f} kg" if "co2_kg_est" in filt.columns and filt["co2_kg_est"].notna().any() else "N/A"

add_kpis_area({
    "Total Orders": f"{total_orders}",
    "On-Time Rate": on_time_rate,
    "Avg Delay": avg_delay,
    "Total CO₂ (est)": co2_total
})

st.divider()

# -----------------------------
# Charts (4+ types)
# -----------------------------
col1, col2 = st.columns(2)

# 1) Trend of on-time rate
if "order_date" in filt.columns and "delayed" in filt.columns and not filt.empty:
    tmp = filt.dropna(subset=["order_date"]).copy()
    tmp["date"] = tmp["order_date"].dt.date
    grp = tmp.groupby("date")["delayed"].apply(lambda s: 100*(1 - s.mean())).reset_index(name="on_time_rate_%")
    with col1:
        st.subheader("📈 On-Time Delivery Trend")
        fig = px.line(grp, x="date", y="on_time_rate_%", markers=True)
        st.plotly_chart(fig, use_container_width=True)

# 2) Delays by carrier
if "carrier" in filt.columns and "delayed" in filt.columns and not filt.empty:
    grp = filt.groupby("carrier")["delayed"].mean().reset_index()
    grp["delay_rate_%"] = 100*grp["delayed"]
    with col2:
        st.subheader("🏷️ Delay Rate by Carrier")
        fig = px.bar(grp, x="carrier", y="delay_rate_%", text_auto=".1f")
        st.plotly_chart(fig, use_container_width=True)

# 3) Distance vs Delay scatter
if {"distance_km","delay_minutes"}.issubset(filt.columns) and not filt.empty:
    with col1:
        st.subheader("🗺️ Distance vs. Delay")
        fig = px.scatter(filt, x="distance_km", y="delay_minutes", hover_data=["order_id","carrier","priority"])
        st.plotly_chart(fig, use_container_width=True)

# 4) Priority mix (pie)
if "priority" in filt.columns and not filt.empty:
    with col2:
        st.subheader("🧩 Order Priority Mix")
        pie = filt["priority"].value_counts().reset_index()
        pie.columns = ["priority","count"]
        fig = px.pie(pie, names="priority", values="count", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# -----------------------------
# ML: Predictive Delay Model
# -----------------------------
st.header("🤖 Predict Delivery Delays")

required_cols = {
    "target": ["delayed"],
    "numeric": ["distance_km","traffic_delay_min","tolls","order_value","co2_kg_est","age_years"],
    "categorical": ["priority","product_category","carrier","origin","destination","vehicle_type"]
}

available_numeric = [c for c in required_cols["numeric"] if c in merged.columns]
available_categ   = [c for c in required_cols["categorical"] if c in merged.columns]

model_df = merged.copy()
model_df = model_df.dropna(subset=["delayed"]) if "delayed" in model_df.columns else pd.DataFrame()

if not model_df.empty and "delayed" in model_df.columns and model_df["delayed"].nunique() > 1:
    X = model_df[available_numeric + available_categ]
    y = model_df["delayed"].astype(int)

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), available_numeric),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), available_categ)
        ],
        remainder="drop"
    )
    clf = RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
    pipe = Pipeline([("pre", pre), ("rf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = float("nan")

    st.subheader("Model Performance")
    add_kpis_area({
        "Accuracy": f"{acc:.3f}",
        "F1-score": f"{f1:.3f}",
        "ROC-AUC": f"{auc:.3f}" if not math.isnan(auc) else "N/A"
    })

    # Feature importance via permutation-like proxy: use RF feature_importances_ and map back
    try:
        rf = pipe.named_steps["rf"]
        oh = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
        num_feats = available_numeric
        cat_feats = list(oh.get_feature_names_out(available_categ)) if available_categ else []
        feat_names = num_feats + cat_feats
        importances = pd.DataFrame({
            "feature": feat_names,
            "importance": rf.feature_importances_[:len(feat_names)]
        }).sort_values("importance", ascending=False).head(15)
        fig = px.bar(importances, x="importance", y="feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Feature importance not available: {e}")

    st.subheader("Order-Level Risk Scoring")
    filt_for_pred = filt.copy()
    if not filt_for_pred.empty:
        # Use same columns as model
        pred_input = filt_for_pred[available_numeric + available_categ].copy()
        try:
            prob = pipe.predict_proba(pred_input)[:,1]
            filt_for_pred["predicted_delay_risk"] = prob
            st.dataframe(filt_for_pred[["order_id","priority","carrier","distance_km","delay_minutes","predicted_delay_risk"]].sort_values("predicted_delay_risk", ascending=False).head(50), use_container_width=True)

            # Download enriched CSV
            csv = filt_for_pred.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Current View with Risk Scores", csv, "orders_with_risk.csv", "text/csv")
        except Exception as e:
            st.warning(f"Could not score current selection: {e}")
else:
    st.info("Not enough labeled data to train the model. Please ensure 'delayed' target is present with both classes.")

st.divider()

# -----------------------------
# Cost & Sustainability Insights
# -----------------------------
st.header("💰 Cost & 🌱 Sustainability Insights")

# Cost leakage: basic rule — high cost & high delay
if {"delivery_cost","delay_minutes"}.issubset(filt.columns) and not filt.empty:
    st.subheader("Cost vs Delay")
    fig = px.scatter(filt, x="delivery_cost", y="delay_minutes", hover_data=["order_id","carrier","priority"])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("Provide 'delivery_cost' and 'delay_minutes' for this analysis.")

# CO2 by carrier
if {"co2_kg_est","carrier"}.issubset(filt.columns) and not filt.empty:
    st.subheader("CO₂ Emissions by Carrier (Estimated)")
    grp = filt.groupby("carrier")["co2_kg_est"].sum().reset_index()
    fig = px.bar(grp, x="carrier", y="co2_kg_est", text_auto=".1f")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Tip: Use the sidebar to upload CSVs or place them in a local 'data/' directory. The app auto-detects files and handles missing columns gracefully.")
