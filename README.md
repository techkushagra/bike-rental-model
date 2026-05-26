# Yolo Rental Bike Fleet Optimization

## Problem Statement

A bike rental company (Yolo Rental Bike) with 4 depots in Bangalore (Bellandur, HSR_Layout, Whitefield, Marathahalli) seeks to maximize weekly profit by optimizing:
- Total fleet size
- Daily rental allocation per depot
- Inter-depot bike transfers
- Repair depot utilization

## Why This Problem Fits S1 (Supply Chain)

Bike rental fleet management is a supply chain optimization problem: physical assets (bikes) flow between locations (depots) to meet spatially distributed demand, subject to capacity constraints (repairs) and logistics costs (transfers). The core challenge—balancing inventory across a network under capacity constraints—is identical to multi-echelon inventory optimization in manufacturing supply chains.


## Model Formulation

The problem is formulated as a deterministic linear programming model assuming steady-state cyclic operation where the days of the week are treated circularly (Monday − 1 = Saturday).

### Indices

| Symbol | Set | Description |
|--------|-----|-------------|
| i, j | {Bellandur, HSR Layout, Whitefield, Marathahalli} | Bike depots |
| t | {Monday, Tuesday, Wednesday, Thursday, Friday, Saturday} | Days |
| k | {1, 2, 3} | Rental duration (days hired) |

### Parameters

| Symbol | Description |
|--------|-------------|
| D_it | Estimated bike rental demand at depot i on day t |
| P_ij | Proportion of bikes rented at depot i returned to depot j |
| C_ij | Transfer cost from depot i to depot j |
| Q_k | Proportion of bikes hired for k days |
| R_i | Repair capacity of depot i |
| RCA_k | Rental revenue for k-day hire returned to same depot |
| RCB_k | Rental revenue for k-day hire returned to different depot |
| RCC | Saturday rental revenue returned Monday to same depot |
| RCD | Saturday rental revenue returned Monday to different depot |
| CS_k | Marginal company cost of a k-day bike rental |

### Decision Variables

| Variable | Description |
|----------|-------------|
| n | Total number of bikes owned |
| nu_it | Undamaged bikes at depot i at beginning of day t |
| nd_it | Damaged bikes at depot i at beginning of day t |
| tr_it | Bikes rented from depot i on day t |
| eu_it | Undamaged idle bikes at depot i during day t |
| ed_it | Damaged idle bikes at depot i during day t |
| tu_ijt | Undamaged bikes transferred from i to j on day t |
| td_ijt | Damaged bikes transferred from i to j on day t |
| rp_it | Damaged bikes repaired at depot i on day t |


## Approach

**Formulation:** Linear Program (LP) with continuous variables, solved with CBC (open-source solver via PuLP).

**Why LP over alternatives:**
- Problem structure is inherently linear (flow balance, capacity bounds)
- Williams confirms integrality gap is negligible ("not worth modelling")
- LP provides dual variables (shadow prices) for sensitivity analysis
- Solves in <0.01s, enabling rapid what-if analysis

**Comparison with:**
1. **MILP** (integer fleet size) — gap is 0.0004%, confirming LP sufficiency
2. **Greedy heuristic** (rule-based, no solver) — LP achieves +24% higher profit

## Project Structure

```
yolo_bike_rental_model/
├── run.py                  # Main orchestrator - run this
├── requirements.txt        # Pinned dependencies
├── README.md               # This file
├── src/
│   ├── __init__.py
│   ├── data.py             # Problem data (Tables 12.20-12.23)
│   ├── lp_model.py         # LP formulation and solver
│   ├── milp_model.py       # MILP variant (integer fleet)
│   ├── heuristic_model.py  # Greedy baseline
│   ├── eda.py              # Exploratory data analysis
│   ├── analysis.py         # Sensitivity, what-if, scalability
│   └── visualize.py        # Result plots
├── results/                # Generated outputs (after running)
│   ├── lp_rentals.csv
│   ├── lp_transfers.csv
│   ├── milp_rentals.csv
│   ├── milp_transfers.csv
│   ├── greedy_rentals.csv
│   ├── greedy_transfers.csv
│   ├── summary.json
│   ├── eda_analysis.png
│   └── results_dashboard.png
└── write-up/
    └── technical_writeup.md
```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python run.py
```

All results (CSVs, plots, summary JSON) are saved to `results/`.

## Key Results

