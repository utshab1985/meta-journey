import pandas as pd
import numpy as np

np.random.seed(42)
n = 2675

# LaLonde characteristics based on original 1986 paper
treatment = np.random.binomial(1, 0.43, n)
age = np.random.normal(25, 7, n)
education = np.random.normal(10, 2, n)
black = np.random.binomial(1, 0.84, n)
married = np.random.binomial(1, 0.19, n)
earnings_1974 = np.maximum(0, np.random.normal(2100, 5600, n))
earnings_1975 = np.maximum(0, np.random.normal(1500, 3200, n))

# True treatment effect = $1794 (LaLonde 1986)
earnings_1978 = (
    1794 * treatment +
    150 * age +
    800 * education +
    500 * black +
    300 * married +
    0.3 * earnings_1974 +
    0.4 * earnings_1975 +
    np.random.normal(0, 3000, n)
)

earnings_1978 = np.maximum(0, earnings_1978)
median_earnings = np.median(earnings_1978)
earned_above_median = (earnings_1978 > median_earnings).astype(int)

df = pd.DataFrame({
    'treatment': treatment,
    'age': age,
    'education': education,
    'black': black,
    'married': married,
    'earnings_1974': earnings_1974,
    'earnings_1975': earnings_1975,
    'earned_above_median': earned_above_median
})

df.to_csv('lalonde.csv', index=False)
print(f"Shape: {df.shape}")
print(f"Treatment rate: {treatment.mean():.1%}")
print(f"Outcome rate: {earned_above_median.mean():.1%}")
print(f"Median earnings threshold: ${median_earnings:,.0f}")
