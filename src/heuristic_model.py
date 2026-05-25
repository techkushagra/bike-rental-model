"""
Greedy heuristic baseline for the Car Rental problem.
No optimization solver - uses simple rule-based allocation.
"""

from copy import deepcopy
from src.data import (
    DEPOTS, DAYS, DURATIONS, NUM_DAYS,
    DEMAND, DURATION_PROP, REPAIR_CAPACITY, REPAIR_DEPOTS,
    OPPORTUNITY_COST, compute_profit_per_rental,
)


def solve_greedy(demand_mult=1.0):
    """
    Rule-based heuristic:
    - Fleet distributed proportionally to total demand per depot
    - Cars rented greedily (first-come-first-served, no prioritization)
    - No inter-depot transfers of undamaged cars
    - Damaged cars sent to nearest repair depot
    - Multi-day rentals properly tracked (cars unavailable until return)

    This represents what a manager might do without optimization software.
    """
    demand = {i: {t: DEMAND[i][t] * demand_mult for t in DAYS} for i in DEPOTS}

    best_profit = float("-inf")
    best_fleet = None
    best_rentals = None
    best_transfers = None

    for fleet_size in range(100, 1500, 25):
        total_demand_by_depot = {i: sum(demand[i].values()) for i in DEPOTS}
        grand_total = sum(total_demand_by_depot.values())
        allocation = {i: max(1, int(fleet_size * total_demand_by_depot[i] / grand_total))
                      for i in DEPOTS}

        # State tracking
        undamaged = {i: float(allocation[i]) for i in DEPOTS}
        damaged_at = {i: 0.0 for i in DEPOTS}
        out_on_rental = {i: [0.0] * NUM_DAYS for i in DEPOTS}

        # Iterate to steady state
        prev_profit = 0.0
        weekly_rentals = {i: {t: 0.0 for t in DAYS} for i in DEPOTS}
        transfers_record = []

        for iteration in range(10):
            weekly_profit = 0.0
            weekly_rentals = {i: {t: 0.0 for t in DAYS} for i in DEPOTS}
            transfers_record = []

            for day_idx in range(NUM_DAYS):
                day = DAYS[day_idx]

                # Returns arrive
                for i in DEPOTS:
                    returning = out_on_rental[i][day_idx]
                    undamaged[i] += returning * 0.9
                    damaged_ret = returning * 0.1
                    if i in REPAIR_DEPOTS:
                        damaged_at[i] += damaged_ret
                    elif i == "Glasgow":
                        damaged_at["Manchester"] += damaged_ret
                        if damaged_ret > 0.5:
                            transfers_record.append((i, "Manchester", day, damaged_ret, "damaged"))
                    else:
                        damaged_at["Birmingham"] += damaged_ret
                        if damaged_ret > 0.5:
                            transfers_record.append((i, "Birmingham", day, damaged_ret, "damaged"))
                    out_on_rental[i][day_idx] = 0.0

                # Repair (limited by capacity)
                for i in REPAIR_DEPOTS:
                    repaired = min(damaged_at[i], float(REPAIR_CAPACITY[i]))
                    damaged_at[i] -= repaired
                    undamaged[i] += repaired

                # Rent greedily
                for i in DEPOTS:
                    available = max(0.0, undamaged[i])
                    rentable = min(available, demand[i][day])
                    if rentable > 0:
                        undamaged[i] -= rentable
                        weekly_rentals[i][day] = rentable
                        weekly_profit += compute_profit_per_rental(i, day) * rentable

                        for k in DURATIONS:
                            num_cars = DURATION_PROP[k] * rentable
                            ret_day_idx = (day_idx + k) % NUM_DAYS
                            out_on_rental[i][ret_day_idx] += num_cars

            weekly_profit -= OPPORTUNITY_COST * fleet_size

            if abs(weekly_profit - prev_profit) < 1.0:
                break
            prev_profit = weekly_profit

        if weekly_profit > best_profit:
            best_profit = weekly_profit
            best_fleet = fleet_size
            best_rentals = deepcopy(weekly_rentals)
            best_transfers = transfers_record[:]

    # Build transfers dict matching LP output format
    transfers_damaged = {}
    transfers_undamaged = {}
    for i in DEPOTS:
        for j in DEPOTS:
            if i != j:
                for t in DAYS:
                    transfers_damaged[(i, j, t)] = 0.0
                    transfers_undamaged[(i, j, t)] = 0.0

    for (src, dst, day, count, _) in best_transfers:
        transfers_damaged[(src, dst, day)] += count

    return {
        "method": "Greedy Heuristic (no optimization)",
        "status": "Heuristic",
        "objective": best_profit,
        "fleet_size": best_fleet,
        "solve_time": 0.0,
        "num_variables": 0,
        "num_constraints": 0,
        "rentals": {(i, t): best_rentals[i][t] for i in DEPOTS for t in DAYS},
        "transfers_undamaged": transfers_undamaged,
        "transfers_damaged": transfers_damaged,
    }
