"""
LP (Linear Programming) formulation for the Car Rental problem.
Continuous relaxation - all variables are continuous.
"""

import time
import pulp
from src.data import (
    DEPOTS, DAYS, DURATIONS, NUM_DEPOTS, NUM_DAYS,
    DEMAND, RETURN_PROP, TRANSFER_COST, DURATION_PROP,
    REPAIR_CAPACITY, OPPORTUNITY_COST,
    compute_profit_per_rental,
)


def build_and_solve_lp(demand_mult=1.0, repair_mult=1.0, transfer_mult=1.0,
                       opportunity_cost=OPPORTUNITY_COST, solver_time_limit=300,
                       verbose=False):
    """
    Build and solve the Car Rental LP.

    Decision Variables:
        n           - total fleet size
        tr[i,t]     - cars rented at depot i on day t
        nu[i,t]     - undamaged cars at depot i, start of day t
        nd[i,t]     - damaged cars at depot i, start of day t
        eu[i,t]     - undamaged cars held idle at depot i on day t
        ed[i,t]     - damaged cars held (not repaired/transferred) at depot i on day t
        tu[i,j,t]   - undamaged cars transferred from i to j on day t
        td[i,j,t]   - damaged cars transferred from i to j on day t
        rp[i,t]     - cars repaired at depot i on day t

    Objective: Maximize weekly profit =
        revenue from rentals + damage surcharge income
        - marginal rental costs - transfer costs - ownership cost

    Constraints:
        1. Undamaged flow balance (in) at each depot/day
        2. Damaged flow balance (in) at each depot/day
        3. Undamaged flow balance (out) at each depot/day
        4. Damaged flow balance (out) at each depot/day
        5. Repair capacity at each depot/day
        6. Demand upper bound at each depot/day
        7. Fleet size counting constraint
    """
    start_time = time.time()

    prob = pulp.LpProblem("Car_Rental_LP", pulp.LpMaximize)

    demand = {i: {t: DEMAND[i][t] * demand_mult for t in DAYS} for i in DEPOTS}
    repair_cap = {i: REPAIR_CAPACITY[i] * repair_mult for i in DEPOTS}
    transfer_cost = {i: {j: TRANSFER_COST[i][j] * transfer_mult for j in DEPOTS} for i in DEPOTS}

    # --- Decision Variables ---
    n = pulp.LpVariable("n_fleet", lowBound=0, cat="Continuous")

    tr = {(i, t): pulp.LpVariable(f"tr_{i}_{t}", lowBound=0)
          for i in DEPOTS for t in DAYS}

    nu = {(i, t): pulp.LpVariable(f"nu_{i}_{t}", lowBound=0)
          for i in DEPOTS for t in DAYS}

    nd = {(i, t): pulp.LpVariable(f"nd_{i}_{t}", lowBound=0)
          for i in DEPOTS for t in DAYS}

    eu = {(i, t): pulp.LpVariable(f"eu_{i}_{t}", lowBound=0)
          for i in DEPOTS for t in DAYS}

    ed = {(i, t): pulp.LpVariable(f"ed_{i}_{t}", lowBound=0)
          for i in DEPOTS for t in DAYS}

    tu = {(i, j, t): pulp.LpVariable(f"tu_{i}_{j}_{t}", lowBound=0)
          for i in DEPOTS for j in DEPOTS for t in DAYS if i != j}

    td = {(i, j, t): pulp.LpVariable(f"td_{i}_{j}_{t}", lowBound=0)
          for i in DEPOTS for j in DEPOTS for t in DAYS if i != j}

    rp = {(i, t): pulp.LpVariable(f"rp_{i}_{t}", lowBound=0)
          for i in DEPOTS for t in DAYS}

    # --- Objective ---
    revenue = pulp.lpSum(
        compute_profit_per_rental(i, t) * tr[i, t]
        for i in DEPOTS for t in DAYS
    )
    transfer_costs_total = pulp.lpSum(
        transfer_cost[i][j] * (tu[i, j, t] + td[i, j, t])
        for i in DEPOTS for j in DEPOTS for t in DAYS if i != j
    )
    ownership = opportunity_cost * n

    prob += revenue - transfer_costs_total - ownership, "Weekly_Profit"

    # --- Constraints ---

    # 1. Undamaged flow IN to depot i on day t
    for i in DEPOTS:
        for t_idx, t in enumerate(DAYS):
            t_prev = DAYS[(t_idx - 1) % NUM_DAYS]

            returns_undamaged = pulp.lpSum(
                0.9 * RETURN_PROP[j][i] * DURATION_PROP[k] * tr[j, DAYS[(t_idx - k) % NUM_DAYS]]
                for j in DEPOTS for k in DURATIONS
            )
            transfers_in = pulp.lpSum(tu[j, i, t_prev] for j in DEPOTS if j != i)

            prob += (returns_undamaged + transfers_in + rp[i, t_prev] + eu[i, t_prev]
                     == nu[i, t], f"undamaged_in_{i}_{t}")

    # 2. Damaged flow IN to depot i on day t
    for i in DEPOTS:
        for t_idx, t in enumerate(DAYS):
            t_prev = DAYS[(t_idx - 1) % NUM_DAYS]

            returns_damaged = pulp.lpSum(
                0.1 * RETURN_PROP[j][i] * DURATION_PROP[k] * tr[j, DAYS[(t_idx - k) % NUM_DAYS]]
                for j in DEPOTS for k in DURATIONS
            )
            transfers_in_d = pulp.lpSum(td[j, i, t_prev] for j in DEPOTS if j != i)

            prob += (returns_damaged + transfers_in_d + ed[i, t_prev]
                     == nd[i, t], f"damaged_in_{i}_{t}")

    # 3. Undamaged flow OUT from depot i on day t
    for i in DEPOTS:
        for t in DAYS:
            transfers_out = pulp.lpSum(tu[i, j, t] for j in DEPOTS if j != i)
            prob += (tr[i, t] + transfers_out + eu[i, t] == nu[i, t],
                     f"undamaged_out_{i}_{t}")

    # 4. Damaged flow OUT from depot i on day t
    for i in DEPOTS:
        for t in DAYS:
            transfers_out_d = pulp.lpSum(td[i, j, t] for j in DEPOTS if j != i)
            prob += (rp[i, t] + transfers_out_d + ed[i, t] == nd[i, t],
                     f"damaged_out_{i}_{t}")

    # 5. Repair capacity
    for i in DEPOTS:
        for t in DAYS:
            prob += (rp[i, t] <= repair_cap[i], f"repair_cap_{i}_{t}")

    # 6. Demand upper bound
    for i in DEPOTS:
        for t in DAYS:
            prob += (tr[i, t] <= demand[i][t], f"demand_{i}_{t}")

    # 7. Fleet size (count cars on Wednesday)
    fleet_expr = pulp.lpSum(
        nu[i, "Wed"] + nd[i, "Wed"]
        + DURATION_PROP[3] * tr[i, "Mon"]
        + (DURATION_PROP[2] + DURATION_PROP[3]) * tr[i, "Tue"]
        for i in DEPOTS
    )
    prob += (fleet_expr == n, "fleet_size")

    # --- Solve ---
    solver = pulp.PULP_CBC_CMD(msg=int(verbose), timeLimit=solver_time_limit)
    prob.solve(solver)
    solve_time = time.time() - start_time

    # --- Extract ---
    result = {
        "method": "LP (Continuous Relaxation)",
        "status": pulp.LpStatus[prob.status],
        "objective": pulp.value(prob.objective) if prob.status == 1 else None,
        "fleet_size": pulp.value(n) if prob.status == 1 else None,
        "solve_time": solve_time,
        "num_variables": prob.numVariables(),
        "num_constraints": prob.numConstraints(),
    }

    if prob.status == 1:
        result["rentals"] = {(i, t): pulp.value(tr[i, t]) for i in DEPOTS for t in DAYS}
        result["transfers_undamaged"] = {
            (i, j, t): pulp.value(tu[i, j, t])
            for i in DEPOTS for j in DEPOTS for t in DAYS if i != j
        }
        result["transfers_damaged"] = {
            (i, j, t): pulp.value(td[i, j, t])
            for i in DEPOTS for j in DEPOTS for t in DAYS if i != j
        }
        result["repairs"] = {(i, t): pulp.value(rp[i, t]) for i in DEPOTS for t in DAYS}

        shadow_prices = {}
        for name, constraint in prob.constraints.items():
            if constraint.pi is not None:
                shadow_prices[name] = constraint.pi
        result["shadow_prices"] = shadow_prices

    return result
