import random
import statistics

# Simulate 1 million users seeing an ad (Meta scale thinking)
random.seed(42)
n_users = 1_000_000

# Each user either saw the ad or didn't
saw_ad = [random.randint(0, 1) for _ in range(n_users)]

# Each user either bought or didn't
# But here's the trick: tech-savvy users buy MORE regardless of the ad
tech_savvy = [random.gauss(0, 1) for _ in range(n_users)]
purchased = [
    1 if (0.3 * saw_ad[i] + 0.7 * tech_savvy[i] + random.gauss(0, 1)) > 1
    else 0
    for i in range(n_users)
]

# Naive analysis (what a bad data scientist does)
buyers_who_saw_ad = [purchased[i] for i in range(n_users) if saw_ad[i] == 1]
buyers_who_didnt = [purchased[i] for i in range(n_users) if saw_ad[i] == 0]

naive_lift = statistics.mean(buyers_who_saw_ad) - statistics.mean(buyers_who_didnt)

print(f"Total users: {n_users:,}")
print(f"Naive ad lift: {naive_lift:.4f}")
print("This number is WRONG. Can you see why?")

from sklearn.linear_model import LogisticRegression
import numpy as np

# Build the input matrix - two columns: saw_ad and tech_savvy
X = np.column_stack([saw_ad, tech_savvy])

# Output - did they buy?
y = np.array(purchased)

# Fit the regression - this finds m1, m2, and c
model = LogisticRegression()
model.fit(X, y)

m1 = model.coef_[0][0]  # ad effect
m2 = model.coef_[0][1]  # tech savvy effect
c  = model.intercept_  # baseline

print(f"\n--- Causal Regression Results ---")
print(f"True ad effect we set:      0.3000")
print(f"Naive measurement found:    0.0766")
print(f"Regression found m1 (ad):   {m1:.4f}")
print(f"Regression found m2 (tech): {m2:.4f}")
print(f"Baseline c:                 {c[0]:.4f}")