"""
Result visualization for the optimization solution.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from src.data import DEPOTS, DAYS, NUM_DEPOTS, NUM_DAYS


def generate_results_dashboard(lp_result, milp_result, greedy_result,
                               sensitivity_results, scalability_results, output_dir):
    """Generate the main results dashboard."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Yolo Rental Bike Fleet Optimization - Results Dashboard", fontsize=14, fontweight="bold")

    # Plot 1: Rental allocation heatmap (LP)
    ax = axes[0, 0]
    rental_data = np.zeros((NUM_DEPOTS, NUM_DAYS))
    for i_idx, i in enumerate(DEPOTS):
        for t_idx, t in enumerate(DAYS):
            rental_data[i_idx, t_idx] = lp_result["rentals"][(i, t)]
    im = ax.imshow(rental_data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(NUM_DAYS))
    ax.set_xticklabels(DAYS)
    ax.set_yticks(range(NUM_DEPOTS))
    ax.set_yticklabels(DEPOTS)
    ax.set_title("LP Optimal Rental Allocation")
    plt.colorbar(im, ax=ax)
    for i_idx in range(NUM_DEPOTS):
        for t_idx in range(NUM_DAYS):
            ax.text(t_idx, i_idx, f"{rental_data[i_idx, t_idx]:.0f}",
                    ha="center", va="center", fontsize=7)

    # Plot 2: Method comparison
    ax = axes[0, 1]
    methods = ["LP", "MILP", "Greedy"]
    profits = [lp_result["objective"], milp_result["objective"], greedy_result["objective"]]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    bars = ax.bar(methods, profits, color=colors)
    ax.set_ylabel("Weekly Profit (INR)")
    ax.set_title("Method Comparison: Weekly Profit")
    for bar, val in zip(bars, profits):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                f"INR {val:,.0f}", ha="center", fontsize=9)

    # Plot 3: Demand sensitivity
    ax = axes[0, 2]
    demand_data = sensitivity_results["demand"]
    x = [d["multiplier"] for d in demand_data]
    y = [d["objective"] for d in demand_data]
    ax.plot(x, y, "bo-", linewidth=2, markersize=6)
    ax.axvline(x=1.0, color="r", linestyle="--", alpha=0.5, label="Base case")
    ax.set_xlabel("Demand Multiplier")
    ax.set_ylabel("Weekly Profit (INR)")
    ax.set_title("Sensitivity: Demand Level")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Repair capacity sensitivity
    ax = axes[1, 0]
    repair_data = sensitivity_results["repair"]
    x = [d["multiplier"] for d in repair_data]
    y = [d["objective"] for d in repair_data]
    ax.plot(x, y, "gs-", linewidth=2, markersize=6)
    ax.axvline(x=1.0, color="r", linestyle="--", alpha=0.5, label="Base case")
    ax.set_xlabel("Repair Capacity Multiplier")
    ax.set_ylabel("Weekly Profit (INR)")
    ax.set_title("Sensitivity: Repair Capacity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Scalability
    ax = axes[1, 1]
    scale_x = [d["num_depots"] for d in scalability_results]
    scale_y = [d["solve_time"] for d in scalability_results]
    ax.plot(scale_x, scale_y, "r^-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Depots")
    ax.set_ylabel("Solve Time (seconds)")
    ax.set_title("Scalability: Solve Time vs Problem Size")
    ax.grid(True, alpha=0.3)

    # Plot 6: Fleet utilization by day
    ax = axes[1, 2]
    fleet = lp_result["fleet_size"]
    utilization = [sum(lp_result["rentals"][(i, t)] for i in DEPOTS) / fleet * 100
                   for t in DAYS]
    ax.bar(DAYS, utilization, color="#4CAF50", alpha=0.8)
    ax.set_ylabel("Fleet Utilization (%)")
    ax.set_title("Daily Fleet Utilization (LP)")
    ax.set_ylim(0, 100)
    mean_util = np.mean(utilization)
    ax.axhline(y=mean_util, color="r", linestyle="--", label=f"Mean: {mean_util:.1f}%")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "results_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dashboard saved to: {path}")
