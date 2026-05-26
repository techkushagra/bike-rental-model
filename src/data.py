"""
Problem data for Yolo Rental Bike Model (adapted from Williams Problem 12.25).
Source: H.P. Williams, "Model Building in Mathematical Programming", 5th Edition.
Tables 12.20, 12.21, 12.22, 12.23 (pages 284-286).

Adapted to a bike rental company with depots in Bangalore.
"""

DEPOTS = ["Bellandur", "HSR_Layout", "Whitefield", "Marathahalli"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
DURATIONS = [1, 2, 3]
NUM_DEPOTS = len(DEPOTS)
NUM_DAYS = len(DAYS)

# Table 12.20: Estimated rental demand (bikes/day)
DEMAND = {
    "Bellandur":    {"Mon": 100, "Tue": 150, "Wed": 135, "Thu": 83,  "Fri": 120, "Sat": 230},
    "HSR_Layout":   {"Mon": 250, "Tue": 143, "Wed": 80,  "Thu": 225, "Fri": 210, "Sat": 98},
    "Whitefield":   {"Mon": 95,  "Tue": 195, "Wed": 242, "Thu": 111, "Fri": 70,  "Sat": 124},
    "Marathahalli": {"Mon": 160, "Tue": 99,  "Wed": 55,  "Thu": 96,  "Fri": 115, "Sat": 80},
}

# Table 12.21: Return proportions - P[origin][destination]
# Probability that a bike rented at origin is returned to destination
RETURN_PROP = {
    "Bellandur":    {"Bellandur": 0.60, "HSR_Layout": 0.20, "Whitefield": 0.10, "Marathahalli": 0.10},
    "HSR_Layout":   {"Bellandur": 0.15, "HSR_Layout": 0.55, "Whitefield": 0.25, "Marathahalli": 0.05},
    "Whitefield":   {"Bellandur": 0.15, "HSR_Layout": 0.20, "Whitefield": 0.54, "Marathahalli": 0.11},
    "Marathahalli": {"Bellandur": 0.08, "HSR_Layout": 0.12, "Whitefield": 0.27, "Marathahalli": 0.53},
}

# Table 12.22: Transfer costs (INR per bike, between depots)
TRANSFER_COST = {
    "Bellandur":    {"Bellandur": 0,  "HSR_Layout": 20, "Whitefield": 30, "Marathahalli": 50},
    "HSR_Layout":   {"Bellandur": 20, "HSR_Layout": 0,  "Whitefield": 15, "Marathahalli": 35},
    "Whitefield":   {"Bellandur": 30, "HSR_Layout": 15, "Whitefield": 0,  "Marathahalli": 25},
    "Marathahalli": {"Bellandur": 50, "HSR_Layout": 35, "Whitefield": 25, "Marathahalli": 0},
}

# Table 12.23: Rental prices (INR) by duration and return type
RENTAL_PRICE_SAME = {1: 50, 2: 70, 3: 120}   # returned to same depot
RENTAL_PRICE_DIFF = {1: 70, 2: 100, 3: 150}   # returned to different depot

# Saturday discount: INR 20 off for hiring on Saturday returned Monday (1-day hire)
SATURDAY_DISCOUNT = 20

# Duration distribution (from past data)
DURATION_PROP = {1: 0.55, 2: 0.20, 3: 0.25}

# Marginal cost to company per rental (wear, tear, administration)
MARGINAL_COST = {1: 20, 2: 25, 3: 30}

# Opportunity cost of owning a bike: INR 15 per bike per week
OPPORTUNITY_COST = 15

# Damage parameters
DAMAGE_RATE = 0.10          # 10% of returned bikes are damaged
DAMAGE_SURCHARGE = 100      # customer charged INR 100 excess

# Repair depot capacities (bikes/day) - only HSR_Layout and Whitefield can repair
REPAIR_CAPACITY = {"Bellandur": 0, "HSR_Layout": 12, "Whitefield": 20, "Marathahalli": 0}
REPAIR_DEPOTS = ["HSR_Layout", "Whitefield"]


def compute_profit_per_rental(depot, day):
    """
    Expected net profit per bike rented at given depot on given day.
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
