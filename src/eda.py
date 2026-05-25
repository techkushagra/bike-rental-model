"""
Exploratory Data Analysis for the Car Rental problem.
Visualizes demand patterns, return flows, and problem structure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.data import (
    DEPOTS, DAYS, DEMAND, RETURN_PROP, TRANSFER_COST,
    REPAIR_CAPACITY, DURATION_PROP, DURATIONS,
    RENTAL_PRICE_SAME, RENTAL_PRICE_DIFF, MARGINAL_COST,
    OPPORTUNITY_COST, DAMAGE_RATE,
)


def run_eda(output_dir):
    """Generate EDA visualizations and print summary statistics."""

    print("\n" + "=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # --- Demand Analysis ---
    demand_df = pd.DataFrame(DEMAND).T
    demand_df.columns = DAYS

    print("\n--- Demand Statistics ---")
    print(f"Total weekly demand: {demand_df.values.sum()} car-rentals")
    print(f"Peak demand: {demand_df.values.max()} (Manchester Monday)")
    print(f"Min demand: {demand_df.values.min()} (Plymouth Wednesday)")
    print(f"Average daily demand per depot: {demand_df.values.mean():.0f}")
    print(f"\nDaily totals:")
    for day in DAYS:
        total = sum(DEMAND[i][day] for i in DEPOTS)
        print(f"  {day}: {total} cars")
    print(f"\nDepot totals (weekly):")
    for depot in DEPOTS:
        total = sum(DEMAND[depot].values())
        print(f"  {depot}: {total} cars ({total/demand_df.values.sum()*100:.1f}%)")

    # --- Flow Network Analysis ---
    print("\n--- Return Flow Analysis ---")
    print("Net flow direction (cars tend to accumulate at):")
    for j in DEPOTS:
        inflow = sum(RETURN_PROP[i][j] for i in DEPOTS) / len(DEPOTS)
        outflow = sum(RETURN_PROP[j][i] for i in DEPOTS) / len(DEPOTS)
        net = inflow - outflow
        direction = "ACCUMULATES" if net > 0 else "DRAINS"
        print(f"  {j:12s}: net flow = {net:+.3f} ({direction})")

    # --- Repair Bottleneck Analysis ---
    print("\n--- Repair Bottleneck Analysis ---")
    total_repair_cap = sum(REPAIR_CAPACITY.values())
    print(f"Total repair capacity: {total_repair_cap} cars/day")
    max_sustainable = total_repair_cap / DAMAGE_RATE
    print(f"Max sustainable rentals (damage equilibrium): {max_sustainable:.0f} cars/day")
    avg_daily_demand = demand_df.values.sum() / len(DAYS)
    print(f"Average daily demand: {avg_daily_demand:.0f} cars/day")
    print(f"Demand vs repair capacity ratio: {avg_daily_demand/max_sustainable:.2f}")
    if avg_daily_demand > max_sustainable:
        print(f"  -> BOTTLENECK: demand exceeds repair throughput by {avg_daily_demand - max_sustainable:.0f} cars/day")
        print(f"  -> Cannot serve all demand in steady state without infinite repair capacity")

    # --- Revenue Analysis ---
    print("\n--- Revenue Structure ---")
    print(f"Duration mix: 1-day={DURATION_PROP[1]*100}%, 2-day={DURATION_PROP[2]*100}%, 3-day={DURATION_PROP[3]*100}%")
    weighted_rev = sum(
        DURATION_PROP[k] * (RENTAL_PRICE_SAME[k] + RENTAL_PRICE_DIFF[k]) / 2
        for k in DURATIONS
    )
    weighted_cost = sum(DURATION_PROP[k] * MARGINAL_COST[k] for k in DURATIONS)
    print(f"Weighted avg revenue per rental: GBP {weighted_rev:.2f}")
    print(f"Weighted avg marginal cost: GBP {weighted_cost:.2f}")
    print(f"Gross margin per rental: GBP {weighted_rev - weighted_cost:.2f}")
    print(f"Break-even fleet utilization (to cover GBP {OPPORTUNITY_COST}/car/week):")
    min_rentals_per_car = OPPORTUNITY_COST / (weighted_rev - weighted_cost)
    print(f"  Need {min_rentals_per_car:.2f} rentals/car/week to break even")

    # --- Generate Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Car Rental Problem - Exploratory Data Analysis", fontsize=13, fontweight="bold")

    # Plot 1: Demand heatmap
    ax = axes[0, 0]
    data = np.array([[DEMAND[i][t] for t in DAYS] for i in DEPOTS])
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(DAYS)))
    ax.set_xticklabels(DAYS)
    ax.set_yticks(range(len(DEPOTS)))
    ax.set_yticklabels(DEPOTS)
    ax.set_title("Demand Heatmap (cars/day)")
    plt.colorbar(im, ax=ax)
    for i in range(len(DEPOTS)):
        for j in range(len(DAYS)):
            ax.text(j, i, str(data[i, j]), ha="center", va="center", fontsize=8)

    # Plot 2: Return flow network
    ax = axes[0, 1]
    positions = {"Glasgow": (0.2, 0.8), "Manchester": (0.8, 0.8),
                 "Birmingham": (0.5, 0.4), "Plymouth": (0.2, 0.1)}
    for depot, (x, y) in positions.items():
        circle = plt.Circle((x, y), 0.08, color="steelblue", alpha=0.7)
        ax.add_patch(circle)
        ax.text(x, y, depot[:4], ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    for i in DEPOTS:
        for j in DEPOTS:
            if i != j:
                xi, yi = positions[i]
                xj, yj = positions[j]
                weight = RETURN_PROP[i][j]
                if weight > 0.10:
                    ax.annotate("", xy=(xj, yj), xytext=(xi, yi),
                               arrowprops=dict(arrowstyle="->", lw=weight * 5, color="gray", alpha=0.6))
                    mx, my = (xi + xj) / 2, (yi + yj) / 2
                    ax.text(mx, my, f"{weight:.0%}", fontsize=7, ha="center")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_title("Return Flow Network (probabilities > 10%)")
    ax.set_aspect("equal")
    ax.axis("off")

    # Plot 3: Daily demand by depot
    ax = axes[1, 0]
    x = np.arange(len(DAYS))
    width = 0.2
    for idx, depot in enumerate(DEPOTS):
        vals = [DEMAND[depot][d] for d in DAYS]
        ax.bar(x + idx * width, vals, width, label=depot, alpha=0.8)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(DAYS)
    ax.set_ylabel("Demand (cars)")
    ax.set_title("Daily Demand by Depot")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Plot 4: Transfer cost matrix
    ax = axes[1, 1]
    cost_data = np.array([[TRANSFER_COST[i][j] for j in DEPOTS] for i in DEPOTS])
    im = ax.imshow(cost_data, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(DEPOTS)))
    ax.set_xticklabels([d[:4] for d in DEPOTS])
    ax.set_yticks(range(len(DEPOTS)))
    ax.set_yticklabels([d[:4] for d in DEPOTS])
    ax.set_title("Transfer Cost Matrix (GBP)")
    plt.colorbar(im, ax=ax)
    for i in range(len(DEPOTS)):
        for j in range(len(DEPOTS)):
            ax.text(j, i, f"£{cost_data[i, j]:.0f}", ha="center", va="center", fontsize=9)

    plt.tight_layout()
    eda_path = os.path.join(output_dir, "eda_analysis.png")
    plt.savefig(eda_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  EDA plots saved to: {eda_path}")

    return {
        "total_weekly_demand": int(demand_df.values.sum()),
        "max_sustainable_rentals_per_day": max_sustainable,
        "avg_daily_demand": avg_daily_demand,
        "repair_bottleneck": avg_daily_demand > max_sustainable,
    }
