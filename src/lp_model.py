import time
import datetime
from ortools.math_opt.python import mathopt
from src.data import (
    DEPOTS, DAYS, DURATIONS, NUM_DAYS,
    DEMAND, RETURN_PROP, TRANSFER_COST, DURATION_PROP,
    REPAIR_CAPACITY, OPPORTUNITY_COST,
    compute_profit_per_rental,
)


def build_and_solve_lp(demand_mult=1.0, repair_mult=1.0, transfer_mult=1.0,
                       opportunity_cost=OPPORTUNITY_COST, solver_time_limit=300):

    start_time = time.time()
    model = mathopt.Model(name="Yolo_Bike_Rental_LP")

    demand = {i: {t: DEMAND[i][t] * demand_mult for t in DAYS} for i in DEPOTS}
    repair_cap = {i: REPAIR_CAPACITY[i] * repair_mult for i in DEPOTS}
    transfer_cost = {i: {j: TRANSFER_COST[i][j] * transfer_mult for j in DEPOTS} for i in DEPOTS}

    # --- Decision Variables ---
    fleet_size = model.add_variable(lb=0.0, name="fleet_size")

    rentals = {(i, t): model.add_variable(lb=0.0, name=f"rentals_{i}_{t}")
               for i in DEPOTS for t in DAYS}

    undamaged_stock = {(i, t): model.add_variable(lb=0.0, name=f"undamaged_stock_{i}_{t}")
                      for i in DEPOTS for t in DAYS}

    damaged_stock = {(i, t): model.add_variable(lb=0.0, name=f"damaged_stock_{i}_{t}")
                    for i in DEPOTS for t in DAYS}

    idle_undamaged = {(i, t): model.add_variable(lb=0.0, name=f"idle_undamaged_{i}_{t}")
                     for i in DEPOTS for t in DAYS}

    idle_damaged = {(i, t): model.add_variable(lb=0.0, name=f"idle_damaged_{i}_{t}")
                   for i in DEPOTS for t in DAYS}

    transfer_undamaged = {(i, j, t): model.add_variable(lb=0.0, name=f"transfer_undamaged_{i}_{j}_{t}")
                         for i in DEPOTS for j in DEPOTS for t in DAYS if i != j}

    transfer_damaged = {(i, j, t): model.add_variable(lb=0.0, name=f"transfer_damaged_{i}_{j}_{t}")
                       for i in DEPOTS for j in DEPOTS for t in DAYS if i != j}

    repairs = {(i, t): model.add_variable(lb=0.0, name=f"repairs_{i}_{t}")
               for i in DEPOTS for t in DAYS}

    # --- Objective ---
    revenue = sum(
        compute_profit_per_rental(i, t) * rentals[i, t]
        for i in DEPOTS for t in DAYS
    )
    transfer_costs_total = sum(
        transfer_cost[i][j] * (transfer_undamaged[i, j, t] + transfer_damaged[i, j, t])
        for i in DEPOTS for j in DEPOTS for t in DAYS if i != j
    )
    ownership = opportunity_cost * fleet_size

    model.maximize(revenue - transfer_costs_total - ownership)

    # --- Constraints ---

    # 1. Undamaged flow IN to depot i on day t
    for i in DEPOTS:
        for t_idx, t in enumerate(DAYS):
            t_prev = DAYS[(t_idx - 1) % NUM_DAYS]

            returns_undamaged = sum(
                0.9 * RETURN_PROP[j][i] * DURATION_PROP[k] * rentals[j, DAYS[(t_idx - k) % NUM_DAYS]]
                for j in DEPOTS for k in DURATIONS
            )
            transfers_in = sum(transfer_undamaged[j, i, t_prev] for j in DEPOTS if j != i)

            model.add_linear_constraint(
                returns_undamaged + transfers_in + repairs[i, t_prev] + idle_undamaged[i, t_prev]
                == undamaged_stock[i, t]
            )

    # 2. Damaged flow IN to depot i on day t
    for i in DEPOTS:
        for t_idx, t in enumerate(DAYS):
            t_prev = DAYS[(t_idx - 1) % NUM_DAYS]

            returns_damaged = sum(
                0.1 * RETURN_PROP[j][i] * DURATION_PROP[k] * rentals[j, DAYS[(t_idx - k) % NUM_DAYS]]
                for j in DEPOTS for k in DURATIONS
            )
            transfers_in_d = sum(transfer_damaged[j, i, t_prev] for j in DEPOTS if j != i)

            model.add_linear_constraint(
                returns_damaged + transfers_in_d + idle_damaged[i, t_prev]
                == damaged_stock[i, t]
            )

    # 3. Undamaged flow OUT from depot i on day t
    for i in DEPOTS:
        for t in DAYS:
            transfers_out = sum(transfer_undamaged[i, j, t] for j in DEPOTS if j != i)
            model.add_linear_constraint(
                rentals[i, t] + transfers_out + idle_undamaged[i, t] == undamaged_stock[i, t]
            )

    # 4. Damaged flow OUT from depot i on day t
    for i in DEPOTS:
        for t in DAYS:
            transfers_out_d = sum(transfer_damaged[i, j, t] for j in DEPOTS if j != i)
            model.add_linear_constraint(
                repairs[i, t] + transfers_out_d + idle_damaged[i, t] == damaged_stock[i, t]
            )

    # 5. Repair capacity
    for i in DEPOTS:
        for t in DAYS:
            model.add_linear_constraint(repairs[i, t] <= repair_cap[i])

    # 6. Demand upper bound
    for i in DEPOTS:
        for t in DAYS:
            model.add_linear_constraint(rentals[i, t] <= demand[i][t])

    # 7. Fleet size (count bikes on Wednesday)
    fleet_expr = sum(
        undamaged_stock[i, "Wed"] + damaged_stock[i, "Wed"]
        + DURATION_PROP[3] * rentals[i, "Mon"]
        + (DURATION_PROP[2] + DURATION_PROP[3]) * rentals[i, "Tue"]
        for i in DEPOTS
    )
    model.add_linear_constraint(fleet_expr == fleet_size)

    # --- Solve ---
    params = mathopt.SolveParameters(time_limit=datetime.timedelta(seconds=solver_time_limit))
    solve_result = mathopt.solve(model, mathopt.SolverType.HIGHS, params=params)
    solve_time = time.time() - start_time

    is_optimal = solve_result.termination.reason == mathopt.TerminationReason.OPTIMAL

    # --- Extract ---
    result = {
        "method": "LP (Continuous Relaxation)",
        "status": "Optimal" if is_optimal else str(solve_result.termination.reason),
        "objective": solve_result.objective_value() if is_optimal else None,
        "fleet_size": solve_result.variable_values()[fleet_size] if is_optimal else None,
        "solve_time": solve_time,
        "num_variables": len(list(model.variables())),
        "num_constraints": len(list(model.linear_constraints())),
    }

    if is_optimal:
        vals = solve_result.variable_values()
        result["rentals"] = {(i, t): vals[rentals[i, t]] for i in DEPOTS for t in DAYS}
        result["transfers_undamaged"] = {
            (i, j, t): vals[transfer_undamaged[i, j, t]]
            for i in DEPOTS for j in DEPOTS for t in DAYS if i != j
        }
        result["transfers_damaged"] = {
            (i, j, t): vals[transfer_damaged[i, j, t]]
            for i in DEPOTS for j in DEPOTS for t in DAYS if i != j
        }
        result["repairs"] = {(i, t): vals[repairs[i, t]] for i in DEPOTS for t in DAYS}

        shadow_prices = {}
        dual_values = solve_result.dual_values()
        for constraint in model.linear_constraints():
            if constraint in dual_values:
                shadow_prices[constraint.name] = dual_values[constraint]
        result["shadow_prices"] = shadow_prices

    return result
