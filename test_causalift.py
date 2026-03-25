import random
import pandas as pd
from causalift import CausalLift

# Generate our familiar 1 million user dataset
random.seed(42)
n_users = 1_000_000

saw_ad = [random.randint(0, 1) for _ in range(n_users)]
tech_savvy = [random.gauss(0, 1) for _ in range(n_users)]
purchased = [
    1 if (0.3 * saw_ad[i] + 0.7 * tech_savvy[i] + random.gauss(0, 1)) > 1
    else 0
    for i in range(n_users)
]

# Put it in a DataFrame - this is how real data arrives
data = pd.DataFrame({
    'saw_ad': saw_ad,
    'tech_savvy': tech_savvy,
    'purchased': purchased
})

# Use CausalLift - three clean lines
model = CausalLift(
    treatment='saw_ad',
    outcome='purchased',
    confounders=['tech_savvy']
)

model.check_assumptions(data)
model.fit(data)
model.ate(data)
model.hte(data)
model.summary()

print("\n--- Predict effect for specific users ---")
high_tech_user = {'tech_savvy': 2.0}
avg_user = {'tech_savvy': 0.0}
low_tech_user = {'tech_savvy': -2.0}

print(f"High tech user effect:     {model.predict_effect(high_tech_user)*100:.1f} per 100")
print(f"Average user effect:       {model.predict_effect(avg_user)*100:.1f} per 100")
print(f"Low tech user effect:      {model.predict_effect(low_tech_user)*100:.1f} per 100")

print("\n\n=== STRESS TEST - Bad Data ===")
import pandas as pd
import numpy as np

np.random.seed(42)
n_small = 50  # tiny dataset

# Severely imbalanced treatment - 95% treated
saw_ad_bad = np.random.binomial(1, 0.95, n_small)
tech_savvy_bad = np.random.normal(0, 1, n_small)
purchased_bad = (0.3 * saw_ad_bad + 
                 0.7 * tech_savvy_bad + 
                 np.random.normal(0, 1, n_small)) > 1
purchased_bad = purchased_bad.astype(int)

df_bad = pd.DataFrame({
    'saw_ad': saw_ad_bad,
    'tech_savvy': tech_savvy_bad,
    'purchased': purchased_bad
})

bad_model = CausalLift(
    treatment='saw_ad',
    outcome='purchased',
    confounders=['tech_savvy']
)

bad_model.check_assumptions(df_bad)