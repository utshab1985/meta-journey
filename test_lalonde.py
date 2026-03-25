import pandas as pd
from causalift import CausalLift

df = pd.read_csv('lalonde.csv')

print("=== CausalLift on LaLonde Dataset ===")
print("Question: Does job training CAUSE higher earnings?")
print(f"Dataset: {len(df)} people, {df['treatment'].mean():.1%} treated")
print()

model = CausalLift(
    treatment='treatment',
    outcome='earned_above_median',
    confounders=['age', 'education', 'black', 
                 'married', 'earnings_1974', 'earnings_1975']
)

model.check_assumptions(df)
model.fit(df)
model.ate(df)
model.hte(df)
model.propensity_score_ate(df)
model.summary()
