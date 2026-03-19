import pandas as pd
from causalift import CausalLift

# Load real data
df = pd.read_csv('hr_data.csv')

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# The causal question:
# Does overtime CAUSE employees to leave?
# Or are stressed people both working overtime AND leaving anyway?

model = CausalLift(
    treatment='overtime',
    outcome='left_company',
    confounders=['stress_score']
)

model.fit(df).summary()