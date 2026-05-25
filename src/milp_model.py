"""
MILP (Mixed-Integer Linear Programming) formulation for the Car Rental problem.
Fleet size is constrained to be integer (whole cars).
"""

import time
import pulp
from src.data import (
    DEPOTS, DAYS, DURATIONS, NUM_DAYS,
    DEMAND, RETURN_PROP, TRANSFER_COST, DURATION_PROP,
    REPAIR_CAPACITY, OPPORTUNITY_COST,
    compute_profit_per_rental,
)


def build_and_solve_milp(solver_time_limit=300, verbose=False):
    """
    Identical to LP but with integer fleet size variable.
    Used to measure the integrality gap and justify LP relaxation.
    """
    start_time = time.time()

    prob = pulp.LpProblem("Car_Rental_MILP", pulp.LpMaximize)

    # Fleet size is the only integer variable
    n = pulp.LpVariable("n_fleet", lowBound=0, cat="Integer")

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

    # Objective
    revenue = pulp.lpSum(
        compute_profit_per_rental(i, t) * tr[i, t]
        for i in DEPOTS for t in DAYS
    )
    transfer_costs = pulp.lpSum(
        TRANSFER_COST[i][j] * (tu[i, j, t] + td[i, j, t])
        for i in DEPOTS for j in DEPOTS for t in DAYS if i != j
    )
    prob += revenue - transfer_costs - OPPORTUNITY_COST * n

    # Constraints (identical to LP)
    for i in DEPOTS:
        for t_idx, t in enumerate(DAYS):
            t_prev = DAYS[(t_idx - 1) % NUM_DAYS]

            returns_u = pulp.lpSum(
                0.9 * RETURN_PROP[j][i] * DURATION_PROP[k] * tr[j, DAYS[(t_idx - k) % NUM_DAYS]]
                for j in DEPOTS for k in DURATIONS
            )
            transfers_in = pulp.lpSum(tu[j, i, t_prev] for j in DEPOTS if j != i)
            prob += (returns_u + transfers_in + rp[i, t_prev] + eu[i, t_prev] == nu[i, t])

            returns_d = pulp.lpSum(
                0.1 * RETURN_PROP[j][i] * DURATION_PROP[k] * tr[j, DAYS[(t_idx - k) % NUM_DAYS]]
                for j in DEPOTS for k in DURATIONS
            )
            transfers_in_d = pulp.lpSum(td[j, i, t_prev] for j in DEPOTS if j != i)
            prob += (returns_d + transfers_in_d + ed[i, t_prev] == nd[i, t])

    for i in DEPOTS:
        for t in DAYS:
            transfers_out = pulp.lpSum(tu[i, j, t] for j in DEPOTS if j != i)
            prob += (tr[i, t] + transfers_out + eu[i, t] == nu[i, t])
            transfers_out_d = pulp.lpSum(td[i, j, t] for j in DEPOTS if j != i)
            prob += (rp[i, t] + transfers_out_d + ed[i, t] == nd[i, t])
            prob += (rp[i, t] <= REPAIR_CAPACITY[i])
            prob += (tr[i, t] <= DEMAND[i][t])

    fleet_expr = pulp.lpSum(
        nu[i, "Wed"] + nd[i, "Wed"]
        + DURATION_PROP[3] * tr[i, "Mon"]
        + (DURATION_PROP[2] + DURATION_PROP[3]) * tr[i, "Tue"]
        for i in DEPOTS
    )
    prob += (fleet_expr == n)

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=int(verbose), timeLimit=solver_time_limit)
    prob.solve(solver)
    solve_time = time.time() - start_time

    result = {
        "method": "MILP (Integer Fleet Size)",
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

    return result
