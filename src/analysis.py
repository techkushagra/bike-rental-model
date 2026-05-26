import time
import datetime
import numpy as np
from ortools.math_opt.python import mathopt
from src.data import (
    DEPOTS, DAYS, NUM_DAYS, DURATIONS, DURATION_PROP,
    DEMAND, RETURN_PROP, TRANSFER_COST, REPAIR_CAPACITY,
    OPPORTUNITY_COST,
)
from src.lp_model import build_and_solve_lp


def interpret_shadow_prices(result):
    """Extract and interpret shadow prices in business terms."""
    if "shadow_prices" not in result:
        return {}

    sp = result["shadow_prices"]

    demand_sp = {}
    for i in DEPOTS:
        for t in DAYS:
            key = f"demand_{i}_{t}"
            if key in sp and abs(sp[key]) > 0.01:
                demand_sp[(i, t)] = sp[key]

    repair_sp = {}
    for i in DEPOTS:
        for t in DAYS:
            key = f"repair_cap_{i}_{t}"
            if key in sp and abs(sp[key]) > 0.01:
                repair_sp[(i, t)] = sp[key]

    fleet_val = sp.get("fleet_size", 0.0)

    return {
        "demand": demand_sp,
        "repair_capacity": repair_sp,
        "fleet_marginal_value": fleet_val,
    }


def run_sensitivity_analysis():
    """Sweep demand, repair capacity, and transfer cost multipliers."""
    results = {"demand": [], "repair": [], "transfer": []}

    for mult in [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]:
        r = build_and_solve_lp(demand_mult=mult)
        results["demand"].append({
            "multiplier": mult,
            "objective": r["objective"],
            "fleet_size": r["fleet_size"],
        })

    for mult in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        r = build_and_solve_lp(repair_mult=mult)
        results["repair"].append({
            "multiplier": mult,
            "objective": r["objective"],
            "fleet_size": r["fleet_size"],
        })

    for mult in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        r = build_and_solve_lp(transfer_mult=mult)
        results["transfer"].append({
            "multiplier": mult,
            "objective": r["objective"],
            "fleet_size": r["fleet_size"],
        })

    return results


def run_what_if_scenarios(base_profit):
    """Test specific business scenarios."""
    scenarios = {}

    # A: HSR_Layout repair depot closed
    orig = REPAIR_CAPACITY["HSR_Layout"]
    REPAIR_CAPACITY["HSR_Layout"] = 0
    r = build_and_solve_lp()
    REPAIR_CAPACITY["HSR_Layout"] = orig
    scenarios["HSR_Layout repair closed"] = {
        "profit": r["objective"],
        "fleet": r["fleet_size"],
        "delta": r["objective"] - base_profit,
        "delta_pct": (r["objective"] - base_profit) / base_profit * 100,
    }

    # B: Double Friday demand
    orig_fri = {i: DEMAND[i]["Fri"] for i in DEPOTS}
    for i in DEPOTS:
        DEMAND[i]["Fri"] *= 2
    r = build_and_solve_lp()
    for i in DEPOTS:
        DEMAND[i]["Fri"] = orig_fri[i]
    scenarios["Double Friday demand"] = {
        "profit": r["objective"],
        "fleet": r["fleet_size"],
        "delta": r["objective"] - base_profit,
        "delta_pct": (r["objective"] - base_profit) / base_profit * 100,
    }

    # C: Transfer costs +50%
    r = build_and_solve_lp(transfer_mult=1.5)
    scenarios["Transfer costs +50%"] = {
        "profit": r["objective"],
        "fleet": r["fleet_size"],
        "delta": r["objective"] - base_profit,
        "delta_pct": (r["objective"] - base_profit) / base_profit * 100,
    }

    # D: Whitefield repair +10 slots
    orig = REPAIR_CAPACITY["Whitefield"]
    REPAIR_CAPACITY["Whitefield"] = 30
    r = build_and_solve_lp()
    REPAIR_CAPACITY["Whitefield"] = orig
    scenarios["Whitefield repair +10"] = {
        "profit": r["objective"],
        "fleet": r["fleet_size"],
        "delta": r["objective"] - base_profit,
        "delta_pct": (r["objective"] - base_profit) / base_profit * 100,
    }

    return scenarios


def run_scalability_analysis():
    """Test solve time scaling with problem size."""
    results = []
    np.random.seed(42)

    for num_depots in [4, 6, 8, 12, 16, 20]:
        depots = [f"D{i}" for i in range(num_depots)]

        model = mathopt.Model(name=f"Scale_{num_depots}")
        n = model.add_variable(lb=0.0, name="n")
        tr = {(i, t): model.add_variable(lb=0.0, name=f"tr_{i}_{t}")
              for i in depots for t in DAYS}
        nu = {(i, t): model.add_variable(lb=0.0, name=f"nu_{i}_{t}")
              for i in depots for t in DAYS}
        eu = {(i, t): model.add_variable(lb=0.0, name=f"eu_{i}_{t}")
              for i in depots for t in DAYS}
        tu = {(i, j, t): model.add_variable(lb=0.0, name=f"tu_{i}_{j}_{t}")
              for i in depots for j in depots for t in DAYS if i != j}
        rp = {(i, t): model.add_variable(lb=0.0, name=f"rp_{i}_{t}")
              for i in depots for t in DAYS}

        model.maximize(sum(50 * tr[i, t] for i in depots for t in DAYS) - 15 * n)

        for i in depots:
            for t_idx, t in enumerate(DAYS):
                t_prev = DAYS[(t_idx - 1) % NUM_DAYS]
                returns = sum(
                    0.9 * (1.0 / num_depots) * 0.55 * tr[j, t_prev] for j in depots
                )
                t_in = sum(tu[j, i, t_prev] for j in depots if j != i)
                model.add_linear_constraint(
                    returns + t_in + eu[i, t_prev] + rp[i, t_prev] == nu[i, t]
                )

                t_out = sum(tu[i, j, t] for j in depots if j != i)
                model.add_linear_constraint(tr[i, t] + t_out + eu[i, t] == nu[i, t])

                model.add_linear_constraint(tr[i, t] <= np.random.randint(50, 200))
                model.add_linear_constraint(rp[i, t] <= (15 if hash(i) % 3 == 0 else 0))

        model.add_linear_constraint(sum(nu[i, "Wed"] for i in depots) == n)

        params = mathopt.SolveParameters(time_limit=datetime.timedelta(seconds=60))
        start = time.time()
        mathopt.solve(model, mathopt.SolverType.HIGHS, params=params)
        elapsed = time.time() - start

        results.append({
            "num_depots": num_depots,
            "num_variables": len(list(model.variables())),
            "num_constraints": len(list(model.linear_constraints())),
            "solve_time": elapsed,
        })

    return results


def production_failure_analysis():
    """Document production failure modes and mitigations."""
    return [
        {
            "failure": "Infeasible model under updated constraints",
            "cause": "Real-time demand or capacity data violates model assumptions "
                     "(e.g., repair depot closed unexpectedly, demand spike beyond fleet capacity)",
            "detection": "Check solver status after each re-solve; alert if status != Optimal",
            "mitigation": "Implement constraint relaxation hierarchy: first relax demand "
                         "(allow unmet demand), then relax repair capacity (allow overflow), "
                         "finally fall back to greedy heuristic",
        },
        {
            "failure": "Solver timeout at scale",
            "cause": "Problem grows beyond HiGHS's practical limit (>50 depots, "
                     ">1000 days in planning horizon)",
            "detection": "Monitor solve time; alert if exceeding SLA (e.g., >30 seconds)",
            "mitigation": "Switch to column generation (decompose by depot) or "
                         "Benders decomposition (separate fleet sizing from allocation). "
                         "For >100 depots, use metaheuristic (simulated annealing).",
        },
        {
            "failure": "Stale input data producing suboptimal solution",
            "cause": "Demand estimates based on historical data that no longer reflects "
                     "reality (seasonal shift, new competitor, pandemic)",
            "detection": "Track actual vs predicted demand weekly; flag >20% deviation",
            "mitigation": "Implement rolling re-estimation with exponential smoothing. "
                         "Re-solve model weekly with updated demand forecasts. "
                         "Maintain demand confidence intervals for robust optimization.",
        },
    ]
