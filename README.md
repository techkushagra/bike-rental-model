# Car Rental Fleet Optimization

**Track C: Applied Optimization | Sub-track: Pure Optimization | Scenario S1: Supply Chain**

## Problem Statement

A car rental company with 4 depots (Glasgow, Manchester, Birmingham, Plymouth) seeks to maximize weekly profit by optimizing:
- Total fleet size
- Daily rental allocation per depot
- Inter-depot car transfers
- Repair depot utilization

This is a **steady-state linear programming** problem where the same fleet distribution repeats each week. The problem comes from H.P. Williams' *"Model Building in Mathematical Programming"* (Problem 12.25, Section 13.25).

## Why This Problem Fits S1 (Supply Chain)

Car rental fleet management is a supply chain optimization problem: physical assets (cars) flow between locations (depots) to meet spatially distributed demand, subject to capacity constraints (repairs) and logistics costs (transfers). The core challenge—balancing inventory across a network under capacity constraints—is identical to multi-echelon inventory optimization in manufacturing supply chains.

## Data Source

**Source:** H.P. Williams, "Model Building in Mathematical Programming", 5th Edition, Tables 12.20–12.23 (pages 284–286).

**Justification:** This is a well-established OR benchmark with realistic scale (4 depots × 6 days × 3 durations), non-trivial constraints (repair capacity bottleneck, stochastic returns, damage flows), and a known optimal structure. The data represents real-world demand patterns with day-of-week seasonality and geographic imbalance.

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
car_rental_model/
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

| Metric | LP Optimal | MILP | Greedy |
|--------|-----------|------|--------|
| Weekly Profit (GBP) | 119,302 | 119,302 | 95,998 |
| Fleet Size | 681 | 682 | 825 |
| Solve Time | 0.01s | 0.02s | N/A |

- **+24% profit improvement** over greedy baseline
- **+50% profit efficiency** per car (GBP 175 vs GBP 116/car/week)
- **144 fewer cars** needed, releasing ~GBP 2.2M capital

## Reproducibility

- Python 3.10+ required
- All dependencies pinned in `requirements.txt`
- Random seed: 42 (for heuristic and scalability tests)
- Single command reproduces all results: `python run.py`
