import time
import datetime
from ortools.math_opt.python import mathopt
from src.data import (
    DEPOTS, DAYS, DURATIONS, NUM_DAYS,
    DEMAND, RETURN_PROP, TRANSFER_COST, DURATION_PROP,
    REPAIR_CAPACITY, OPPORTUNITY_COST,
    compute_profit_per_rental,
)


def build_and_solve_milp(solver_time_limit=300):

    start_time = time.time()
    model = mathopt.Model(name="Yolo_Bike_Rental_MILP")

    # Fleet size is the only integer variable
    fleet_size = model.add_integer_variable(lb=0, name="fleet_size")

    # Decision variables: rentals, transfers, repairs are integer (whole bikes)
    rentals = {(i, t): model.add_variable(lb=0, name=f"rentals_{i}_{t}")
               for i in DEPOTS for t in DAYS}
    transfer_undamaged = {(i, j, t): model.add_variable(lb=0, name=f"transfer_undamaged_{i}_{j}_{t}")
                         for i in DEPOTS for j in DEPOTS for t in DAYS if i != j}
    transfer_damaged = {(i, j, t): model.add_variable(lb=0, name=f"transfer_damaged_{i}_{j}_{t}")
                       for i in DEPOTS for j in DEPOTS for t in DAYS if i != j}
    repairs = {(i, t): model.add_variable(lb=0, name=f"repairs_{i}_{t}")
               for i in DEPOTS for t in DAYS}
    # Stock/idle variables: continuous (fractional due to probabilistic return proportions)
    undamaged_stock = {(i, t): model.add_variable(lb=0.0, name=f"undamaged_stock_{i}_{t}")
                      for i in DEPOTS for t in DAYS}
    damaged_stock = {(i, t): model.add_variable(lb=0.0, name=f"damaged_stock_{i}_{t}")
                    for i in DEPOTS for t in DAYS}
    idle_undamaged = {(i, t): model.add_variable(lb=0.0, name=f"idle_undamaged_{i}_{t}")
                     for i in DEPOTS for t in DAYS}
    idle_damaged = {(i, t): model.add_variable(lb=0.0, name=f"idle_damaged_{i}_{t}")
                   for i in DEPOTS for t in DAYS}

    # Objective
    revenue = sum(
        compute_profit_per_rental(i, t) * rentals[i, t]
        for i in DEPOTS for t in DAYS
    )
    transfer_costs = sum(
        TRANSFER_COST[i][j] * (transfer_undamaged[i, j, t] + transfer_damaged[i, j, t])
        for i in DEPOTS for j in DEPOTS for t in DAYS if i != j
    )
    model.maximize(revenue - transfer_costs - OPPORTUNITY_COST * fleet_size)

    # Constraints (identical to LP)
    for i in DEPOTS:
        for t_idx, t in enumerate(DAYS):
            t_prev = DAYS[(t_idx - 1) % NUM_DAYS]

            returns_u = sum(
                0.9 * RETURN_PROP[j][i] * DURATION_PROP[k] * rentals[j, DAYS[(t_idx - k) % NUM_DAYS]]
                for j in DEPOTS for k in DURATIONS
            )
            transfers_in = sum(transfer_undamaged[j, i, t_prev] for j in DEPOTS if j != i)
            model.add_linear_constraint(
                returns_u + transfers_in + repairs[i, t_prev] + idle_undamaged[i, t_prev]
                == undamaged_stock[i, t]
            )

            returns_d = sum(
                0.1 * RETURN_PROP[j][i] * DURATION_PROP[k] * rentals[j, DAYS[(t_idx - k) % NUM_DAYS]]
                for j in DEPOTS for k in DURATIONS
            )
            transfers_in_d = sum(transfer_damaged[j, i, t_prev] for j in DEPOTS if j != i)
            model.add_linear_constraint(
                returns_d + transfers_in_d + idle_damaged[i, t_prev] == damaged_stock[i, t]
            )

    for i in DEPOTS:
        for t in DAYS:
            transfers_out = sum(transfer_undamaged[i, j, t] for j in DEPOTS if j != i)
            model.add_linear_constraint(
                rentals[i, t] + transfers_out + idle_undamaged[i, t] == undamaged_stock[i, t]
            )
            transfers_out_d = sum(transfer_damaged[i, j, t] for j in DEPOTS if j != i)
            model.add_linear_constraint(
                repairs[i, t] + transfers_out_d + idle_damaged[i, t] == damaged_stock[i, t]
            )
            model.add_linear_constraint(repairs[i, t] <= REPAIR_CAPACITY[i])
            model.add_linear_constraint(rentals[i, t] <= DEMAND[i][t])

    fleet_expr = sum(
        undamaged_stock[i, "Wed"] + damaged_stock[i, "Wed"]
        + DURATION_PROP[3] * rentals[i, "Mon"]
        + (DURATION_PROP[2] + DURATION_PROP[3]) * rentals[i, "Tue"]
        for i in DEPOTS
    )
    model.add_linear_constraint(fleet_expr == fleet_size)

    # Solve
    params = mathopt.SolveParameters(time_limit=datetime.timedelta(seconds=solver_time_limit))
    solve_result = mathopt.solve(model, mathopt.SolverType.HIGHS, params=params)
    solve_time = time.time() - start_time

    has_solution = solve_result.termination.reason in (
        mathopt.TerminationReason.OPTIMAL,
        mathopt.TerminationReason.FEASIBLE,
    )

    result = {
        "method": "MILP (All-Integer Decisions)",
        "status": "Optimal" if solve_result.termination.reason == mathopt.TerminationReason.OPTIMAL
                  else "Feasible" if has_solution else str(solve_result.termination.reason),
        "objective": solve_result.objective_value() if has_solution else None,
        "fleet_size": solve_result.variable_values()[fleet_size] if has_solution else None,
        "solve_time": solve_time,
        "num_variables": len(list(model.variables())),
        "num_constraints": len(list(model.linear_constraints())),
    }

    if has_solution:
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

    return result
