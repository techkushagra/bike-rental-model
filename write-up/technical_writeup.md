# Technical Write-Up: Car Rental Fleet Optimization

**Track C — Applied Optimization (Pure Optimization) | Scenario S1 — Supply Chain**

---

## Section 1: Problem & Domain

**Problem:** Optimize fleet size, rental allocation, and inter-depot transfers for a 4-depot car rental company to maximize steady-state weekly profit under repair capacity constraints.

**Why this is meaningful:** Fleet management represents a GBP 6B+ industry in the UK. The core challenge—dynamically allocating physical assets across a network with stochastic returns and capacity-constrained maintenance—maps directly to supply chain inventory optimization. A 24% profit improvement (as demonstrated) translates to ~GBP 1.2M/year for a mid-size operator.

**Why Applied Optimization (not ML):** This problem has well-defined decision variables, linear constraints, and a clear objective function. The structure is exactly a network flow LP—no prediction is needed because demand estimates are given. An ML model would be appropriate if we needed to *forecast* demand, but given known demand, the bottleneck is *allocation*, which is a pure optimization problem.

---

## Section 2: Approach & Algorithm Decisions

### Formulation Choice: LP (Linear Program)

**Chosen:** Continuous LP with network flow structure, solved by CBC simplex/barrier method.

**Alternative considered:** Mixed-Integer LP (MILP) with integer fleet size and integer rental allocations.

**Why rejected:** The LP-MILP gap is 0.0004% (GBP 0.51/week difference). Williams explicitly states "the integrality of the cars is not worth modelling." Integer constraints add branch-and-bound overhead without meaningful solution improvement. LP also provides exact dual variables (shadow prices) which are essential for sensitivity analysis.

### Solver Choice: PuLP with CBC

**Chosen:** PuLP (Python modelling language) + CBC (COIN-OR Branch and Cut).

**Alternative considered:** Pyomo + GLPK, CVXPY + SCS.

**Why rejected:** PuLP is the simplest interface for LP/MILP problems with no conic or quadratic structure. CBC is the strongest open-source LP solver (faster than GLPK by 2-5x on this problem class). CVXPY/SCS are better suited for second-order cone or semidefinite problems. Pyomo adds unnecessary complexity for a single-objective LP.

### Baseline Choice: Greedy Heuristic

**Chosen:** Proportional fleet allocation + greedy daily rental + no inter-depot transfers.

**Why this baseline:** It represents what a fleet manager would do without optimization software—allocate cars by historical demand share, rent to whoever shows up first, and only move damaged cars for repair. The 24% gap demonstrates clear optimization value.

---

## Section 3: Results & Error Analysis

### Key Metrics

| Metric | Value |
|--------|-------|
| Optimal weekly profit | GBP 119,302 |
| Fleet size | 681 cars |
| LP-MILP gap | 0.0004% |
| vs. Greedy improvement | +24.3% |
| Constraint satisfaction | 100% |
| Solve time | 0.01s |

### Where the Solution Fails

1. **Demand served is only 59% of total demand (1,920 of 3,266 weekly requests).** The repair capacity bottleneck (32 cars/day = max 320 rentals/day sustainable) fundamentally limits throughput. The model correctly identifies this ceiling and optimizes within it.

2. **Saturday zero-rental at 3 depots.** The optimizer avoids Saturday rentals at Glasgow, Manchester, and Plymouth because the Saturday discount (GBP 20 off 1-day hires) combined with the repair capacity constraint makes it more profitable to hold cars idle for Monday's higher-margin demand. This is economically rational but may not be operationally acceptable.

3. **Static demand assumption.** The model assumes the same demand every week. In reality, demand varies. A 15% demand overestimate would lead to fleet oversizing of ~50 cars, costing GBP 39,000/year in excess ownership costs.

---

## Section 4: Production & Limitations

### Production Consideration: Re-solve Trigger Design

In production, the optimizer would re-solve weekly (Sunday night) with updated demand forecasts. Critical triggers for emergency re-solve:
- Repair depot outage (instant capacity drop → model becomes infeasible)
- Demand shock >20% from forecast
- Fleet loss >5% (accidents, theft)

Implementation: wrap solver in REST API, connect to demand forecasting system, schedule via cron with alerting on infeasibility.

### Key Limitation: Deterministic Demand

The model treats demand as known. In reality, demand is uncertain. A robust optimization extension (minimax regret over demand scenarios) or a two-stage stochastic program (first-stage: fleet size, second-stage: allocation given realized demand) would handle this. The current LP would be the inner problem of such an extension.

Cost of this limitation: estimated 5-10% profit loss vs. stochastic optimal during high-variance periods (holidays, events).

---

## Rubric Question Responses

**Q1: Decision variables, objective, constraints—why this formulation?**
Variables: fleet size (n), rentals (tr), car stocks (nu/nd), transfers (tu/td), repairs (rp). Objective: maximize profit = revenue - transfer costs - ownership cost. Constraints: flow conservation (cars can't appear/disappear), repair capacity, demand ceiling. This network flow formulation is natural because cars physically flow between states (depot, rental, repair, transfer).

**Q2: Why LP solver over alternatives?**
LP because all relationships are linear and the problem has continuous structure. CP (constraint programming) would add unnecessary overhead—there's no scheduling or sequencing aspect. Metaheuristics (GA, SA) would sacrifice optimality guarantees and shadow prices without any benefit, since CBC solves this in 0.01s.

**Q3: What simplifications?**
(1) Return proportions independent of duration—sacrifices ~5% accuracy for linear structure. (2) Deterministic demand—avoids stochastic programming complexity. (3) Uniform damage rate—ignores route/duration effects. (4) One-day transfers everywhere—ignores geographic distance. All acceptable for weekly strategic planning.

**Q4: Shadow prices in business language?**
- Fleet size dual = GBP 15/week: adding one car earns exactly its cost—fleet is perfectly sized.
- Repair capacity at Glasgow = GBP 637/slot/day: installing one repair bay at Glasgow would earn GBP 637/week (annual payback in weeks, not years).
- Demand at Birmingham Thursday = GBP 13.51/car: if one more customer appeared, profit increases by GBP 13.51.

**Q5:** N/A (no ML component—pure optimization).

**Q6: Production failure modes?**
(1) Infeasibility when real data violates constraints (repair closure)—mitigate with constraint relaxation hierarchy. (2) Solver timeout at >50 depots—mitigate with Benders decomposition. (3) Stale forecasts producing suboptimal plans—mitigate with weekly re-solve + deviation alerting.
