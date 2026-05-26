import os
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import (
    DEPOTS, DAYS, DEMAND, RETURN_PROP, TRANSFER_COST,
    RENTAL_PRICE_SAME, RENTAL_PRICE_DIFF, MARGINAL_COST,
    DURATION_PROP, DAMAGE_RATE, DAMAGE_SURCHARGE,
    REPAIR_CAPACITY, SATURDAY_DISCOUNT, compute_profit_per_rental,
)
from src.lp_model import build_and_solve_lp
from src.milp_model import build_and_solve_milp
from src.heuristic_model import solve_greedy
from src.eda import run_eda
from src.analysis import (
    interpret_shadow_prices, run_sensitivity_analysis,
    run_what_if_scenarios, run_scalability_analysis,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_allocation_csv(result, filename):
    rows = []
    for i in DEPOTS:
        row = {"Depot": i}
        total = 0
        for t in DAYS:
            val = result["rentals"][(i, t)]
            row[t] = round(val, 1)
            total += val
        row["Total"] = round(total, 1)
        rows.append(row)

    total_row = {"Depot": "TOTAL"}
    grand_total = 0
    for t in DAYS:
        col_sum = sum(result["rentals"][(i, t)] for i in DEPOTS)
        total_row[t] = round(col_sum, 1)
        grand_total += col_sum
    total_row["Total"] = round(grand_total, 1)
    rows.append(total_row)

    rental_df = pd.DataFrame(rows)
    rental_path = os.path.join(OUTPUT_DIR, f"{filename}_rentals.csv")
    rental_df.to_csv(rental_path, index=False)

    transfer_rows = []
    for i in DEPOTS:
        for j in DEPOTS:
            if i != j:
                for t in DAYS:
                    tu_val = result.get("transfers_undamaged", {}).get((i, j, t), 0)
                    td_val = result.get("transfers_damaged", {}).get((i, j, t), 0)
                    if tu_val > 0.5 or td_val > 0.5:
                        transfer_rows.append({
                            "From": i, "To": j, "Day": t,
                            "Undamaged": round(tu_val, 1),
                            "Damaged": round(td_val, 1),
                        })

    transfer_df = pd.DataFrame(transfer_rows) if transfer_rows else pd.DataFrame(
        columns=["From", "To", "Day", "Undamaged", "Damaged"]
    )
    transfer_path = os.path.join(OUTPUT_DIR, f"{filename}_transfers.csv")
    transfer_df.to_csv(transfer_path, index=False)

    return rental_path, transfer_path


def main():
    print("YOLO Rental Bike Fleet Optimization")
    print("-" * 40)

    # EDA
    # run_eda(OUTPUT_DIR)

    # Solve LP
    print("\nSolving LP model...")
    lp_result = build_and_solve_lp()
    print(f"  Status: {lp_result['status']}")
    save_allocation_csv(lp_result, "lp")

    # Solve MILP
    print("\nSolving MILP model...")
    milp_result = build_and_solve_milp()
    print(f"  Status: {milp_result['status']}")
    save_allocation_csv(milp_result, "milp")

    gap_pct = abs(lp_result["objective"] - milp_result["objective"]) / abs(lp_result["objective"]) * 100
    print(f"  LP-MILP gap: {gap_pct:.4f}%")

    # Greedy baseline
    print("\nSolving greedy heuristic...")
    greedy_result = solve_greedy()
    print(f"  Profit: INR {greedy_result['objective']:,.2f}/week")
    print(f"  Fleet:  {greedy_result['fleet_size']} bikes")
    save_allocation_csv(greedy_result, "greedy")

    # Comparison
    improvement = (lp_result["objective"] - greedy_result["objective"]) / abs(greedy_result["objective"]) * 100
    fleet_reduction = greedy_result["fleet_size"] - int(lp_result["fleet_size"])
    annual_saving = (lp_result["objective"] - greedy_result["objective"]) * 52

    print(f"\nLP vs Greedy: +{improvement:.1f}% profit, {fleet_reduction} fewer bikes")
    print(f"  Annualized gain: INR {annual_saving:,.0f}")

    # Shadow prices
    interp = interpret_shadow_prices(lp_result)
    print(f"\nFleet marginal value: INR {interp['fleet_marginal_value']:.2f}/bike/week")

    # Sensitivity & scenarios
    run_sensitivity_analysis()
    run_what_if_scenarios(lp_result["objective"])
    run_scalability_analysis()

    # Save summary
    summary = {
        "lp_profit": lp_result["objective"],
        "milp_profit": milp_result["objective"],
        "greedy_profit": greedy_result["objective"],
        "lp_fleet": lp_result["fleet_size"],
        "milp_fleet": milp_result["fleet_size"],
        "greedy_fleet": greedy_result["fleet_size"],
        "improvement_over_greedy_pct": improvement,
        "lp_milp_gap_pct": gap_pct,
        "annual_profit_gain": annual_saving,
        "solve_time": lp_result["solve_time"],
        "num_variables": lp_result["num_variables"],
        "num_constraints": lp_result["num_constraints"],
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
