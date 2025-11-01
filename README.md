# 🚚 NexGen Logistics - Predictive Delivery Optimizer

A Machine Learning–powered **Streamlit dashboard** that predicts delivery delays, analyzes cost leakage, and tracks environmental impact to help NexGen Logistics transform into a **data-driven, predictive** logistics organization.

---

## 🏢 Business Context

NexGen Logistics is facing the following challenges:
- Increasing **delivery delays**
- Rising **logistics cost** and fuel usage
- Limited visibility into **fleet & route performance**
- **Customer dissatisfaction** due to SLA breaches

✅ Objective: Enhance delivery performance by *predicting delays before they happen*  
✅ Approach: Data analytics + ML + visualization  
✅ Value: Better planning, lower risk, improved customer satisfaction

---

## ✅ Key Features & Benefits

| Feature | Benefit |
|--------|---------|
| 📊 Operational KPIs | Instant visibility of performance |
| 📈 4+ Interactive Analytics Charts | Identify bottlenecks & trends |
| 🤖 ML Delay Prediction | React early and prevent late deliveries |
| 🚨 Risk Scoring Table | Sort high-risk orders & download CSV |
| 🌱 CO₂ Emission Estimation | Measure environmental footprint |
| 💰 Cost Insights | Detect high-cost + high-delay orders |
| 🎛 Smart Filters | Slice data by priority, routes, carriers etc. |
| 🧱 Schema Tolerant | Works even if some columns are missing |

---

## 🧠 Machine Learning Approach

| Attribute | Details |
|---------|---------|
| Model Used | RandomForestClassifier |
| Training Strategy | Train/Test split (75/25) |
| Target Label | `delayed` (automatically derived) |
| Metrics Shown | Accuracy, F1-Score, ROC-AUC |
| Feature Engineering | Priority, product category, delays, CO₂ estimate, carrier, vehicle type etc. |
| Evaluation Result | Displayed inside dashboard |

⚙️ Missing data handled with **imputation** & preprocessing pipeline ✅

---

## 📂 Dataset Description (7 CSV files)

Store inside `data/` folder:

| File Name | Description |
|----------|-------------|
| `orders.csv` | Order details: date, priority, product, origin, destination |
| `delivery_performance.csv` | Promised vs actual time, carrier, status, cost |
| `routes_distance.csv` | Distance, tolls, traffic delay, weather impact |
| `vehicle_fleet.csv` | Vehicle type, capacity, CO₂ data, age |
| `warehouse_inventory.csv` | Stock information across warehouses |
| `customer_feedback.csv` | Ratings, issue categories |
| `cost_breakdown.csv` | Cost components per order |

> ✅ Realistic values and relationships  
> ✅ Incomplete rows allowed (tool handles missing values)

---

## 🧩 Project Structure

nexgen_logistics_optimizer/
│
├── app.py
├── requirements.txt
├── README.md
├── Innovation_Brief.pdf
│
└── data/
├── orders.csv
├── delivery_performance.csv
├── routes_distance.csv
├── vehicle_fleet.csv
├── warehouse_inventory.csv
├── customer_feedback.csv
└── cost_breakdown.csv


---

## 🔧 Installation & How to Run

### ✅ Prerequisites
✔ Python 3.8+  
✔ Windows/Linux/Mac  
✔ Browser (Chrome recommended)

### 🔽 Commands

```bash
# Go into project folder
cd nexgen_logistics_optimizer

# Create virtual environment
python -m venv .venv

# Activate venv (Windows)
.venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py

Screenshots: (Dashboard & visualizations) All The screenshots ar in the folder (>screenshots)

