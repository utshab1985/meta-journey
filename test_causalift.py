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

model.fit(data).summary()