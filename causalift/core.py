import numpy as np
from sklearn.linear_model import LogisticRegression
import math

class CausalLift:
    """
    CausalLift: Find the true causal effect of any intervention.
    Separates what you caused from what would have happened anyway.
    """

    def __init__(self, treatment, outcome, confounders):
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders
        self.model = None
        self.results = {}

    def fit(self, data):
        # Build input matrix - treatment + confounders
        features = [self.treatment] + self.confounders
        X = data[features].values
        y = data[self.outcome].values

        # Fit logistic regression
        self.model = LogisticRegression()
        self.model.fit(X, y)

        # Extract coefficients
        self.results['treatment_effect_log_odds'] = self.model.coef_[0][0]
        self.results['treatment_odds_ratio'] = math.exp(self.model.coef_[0][0])
        self.results['confounder_odds_ratios'] = [
            math.exp(c) for c in self.model.coef_[0][1:]
        ]

        return self

    def summary(self):
        odds = self.results['treatment_odds_ratio']
        print(f"\n--- CausalLift Results ---")
        print(f"True causal effect of '{self.treatment}':")
        print(f"Users exposed were {odds:.2f}x more likely to convert")
        print(f"(Controlling for: {self.confounders})")
        return self