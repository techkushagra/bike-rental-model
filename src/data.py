"""
Problem data for Car Rental 1 (Williams Problem 12.25).
Source: H.P. Williams, "Model Building in Mathematical Programming", 5th Edition.
Tables 12.20, 12.21, 12.22, 12.23 (pages 284-286).
"""

DEPOTS = ["Glasgow", "Manchester", "Birmingham", "Plymouth"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
DURATIONS = [1, 2, 3]
NUM_DEPOTS = len(DEPOTS)
NUM_DAYS = len(DAYS)

# Table 12.20: Estimated rental demand (cars/day)
DEMAND = {
    "Glasgow":    {"Mon": 100, "Tue": 150, "Wed": 135, "Thu": 83,  "Fri": 120, "Sat": 230},
    "Manchester": {"Mon": 250, "Tue": 143, "Wed": 80,  "Thu": 225, "Fri": 210, "Sat": 98},
    "Birmingham": {"Mon": 95,  "Tue": 195, "Wed": 242, "Thu": 111, "Fri": 70,  "Sat": 124},
    "Plymouth":   {"Mon": 160, "Tue": 99,  "Wed": 55,  "Thu": 96,  "Fri": 115, "Sat": 80},
}

# Table 12.21: Return proportions - P[origin][destination]
# Probability that a car rented at origin is returned to destination
RETURN_PROP = {
    "Glasgow":    {"Glasgow": 0.60, "Manchester": 0.20, "Birmingham": 0.10, "Plymouth": 0.10},
    "Manchester": {"Glasgow": 0.15, "Manchester": 0.55, "Birmingham": 0.25, "Plymouth": 0.05},
    "Birmingham": {"Glasgow": 0.15, "Manchester": 0.20, "Birmingham": 0.54, "Plymouth": 0.11},
    "Plymouth":   {"Glasgow": 0.08, "Manchester": 0.12, "Birmingham": 0.27, "Plymouth": 0.53},
}

# Table 12.22: Transfer costs (GBP per car, between depots)
TRANSFER_COST = {
    "Glasgow":    {"Glasgow": 0,  "Manchester": 20, "Birmingham": 30, "Plymouth": 50},
    "Manchester": {"Glasgow": 20, "Manchester": 0,  "Birmingham": 15, "Plymouth": 35},
    "Birmingham": {"Glasgow": 30, "Manchester": 15, "Birmingham": 0,  "Plymouth": 25},
    "Plymouth":   {"Glasgow": 50, "Manchester": 35, "Birmingham": 25, "Plymouth": 0},
}

# Table 12.23: Rental prices (GBP) by duration and return type
RENTAL_PRICE_SAME = {1: 50, 2: 70, 3: 120}   # returned to same depot
RENTAL_PRICE_DIFF = {1: 70, 2: 100, 3: 150}   # returned to different depot

# Saturday discount: GBP 20 off for hiring on Saturday returned Monday (1-day hire)
SATURDAY_DISCOUNT = 20

# Duration distribution (from past data)
DURATION_PROP = {1: 0.55, 2: 0.20, 3: 0.25}

# Marginal cost to company per rental (wear, tear, administration)
MARGINAL_COST = {1: 20, 2: 25, 3: 30}

# Opportunity cost of owning a car: GBP 15 per car per week
OPPORTUNITY_COST = 15

# Damage parameters
DAMAGE_RATE = 0.10          # 10% of returned cars are damaged
DAMAGE_SURCHARGE = 100      # customer charged GBP 100 excess

# Repair depot capacities (cars/day) - only Manchester and Birmingham can repair
REPAIR_CAPACITY = {"Glasgow": 0, "Manchester": 12, "Birmingham": 20, "Plymouth": 0}
REPAIR_DEPOTS = ["Manchester", "Birmingham"]


def compute_profit_per_rental(depot, day):
    """
    Expected net profit per car rented at given depot on given day.
    Accounts for: rental price, marginal cost, damage surcharge income,
    duration mix, same/different depot return probabilities, Saturday discount.
    """
    p_same = RETURN_PROP[depot][depot]
    p_diff = 1.0 - p_same
    damage_income = DAMAGE_RATE * DAMAGE_SURCHARGE

    profit = 0.0
    for k in DURATIONS:
        if day == "Sat" and k == 1:
            rev_same = RENTAL_PRICE_SAME[1] - SATURDAY_DISCOUNT
            rev_diff = RENTAL_PRICE_DIFF[1] - SATURDAY_DISCOUNT
        else:
            rev_same = RENTAL_PRICE_SAME[k]
            rev_diff = RENTAL_PRICE_DIFF[k]

        net_same = rev_same - MARGINAL_COST[k] + damage_income
        net_diff = rev_diff - MARGINAL_COST[k] + damage_income

        profit += DURATION_PROP[k] * (p_same * net_same + p_diff * net_diff)

    return profit
