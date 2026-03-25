import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score
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

    def check_assumptions(self, data):
        import numpy as np
        
        print("\n--- CausalLift Assumption Checks ---")
        all_passed = True
        
        # Check 1: Sample size
        n_rows = len(data)
        n_params = 1 + len(self.confounders)
        ratio = n_rows / n_params
        
        if ratio < 100:
            print(f"WARNING: Sample size may be insufficient")
            print(f"  Rows: {n_rows}, Parameters: {n_params}, Ratio: {ratio:.0f}")
            print(f"  Recommended: at least 100 rows per parameter")
            all_passed = False
        else:
            print(f"Sample size:      PASSED ({n_rows:,} rows, ratio {ratio:.0f}:1)")
        
        # Check 2: Treatment balance
        treatment_rate = data[self.treatment].mean()
        
        if treatment_rate < 0.1 or treatment_rate > 0.9:
            print(f"WARNING: Treatment is severely imbalanced")
            print(f"  {treatment_rate*100:.1f}% treated, {(1-treatment_rate)*100:.1f}% control")
            print(f"  Causal estimates may be unreliable")
            all_passed = False
        elif treatment_rate < 0.2 or treatment_rate > 0.8:
            print(f"Treatment balance: WARNING ({treatment_rate*100:.1f}% treated)")
            print(f"  Mild imbalance — interpret results with caution")
        else:
            print(f"Treatment balance: PASSED ({treatment_rate*100:.1f}% treated)")
        
        # Check 3: Confounder relevance
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        X_conf = data[self.confounders].values
        X_treat = data[self.treatment].values
        conf_model = LinearRegression()
        conf_model.fit(X_conf, X_treat)
        r2 = r2_score(X_treat, conf_model.predict(X_conf))
        
        if r2 < 0.01:
            print(f"Confounder relevance: LOW (R²={r2:.4f})")
            print(f"  Confounders barely predict treatment")
            print(f"  Consider whether these are true confounders")
        else:
            print(f"Confounder relevance: PASSED (R²={r2:.4f})")
        
        # Check 4: Overlap assumption
        treated = data[data[self.treatment] == 1]
        control = data[data[self.treatment] == 0]
        
        overlap_violations = 0
        for confounder in self.confounders:
            treated_range = (
                treated[confounder].min(),
                treated[confounder].max()
            )
            control_range = (
                control[confounder].min(),
                control[confounder].max()
            )
            
            overlap_min = max(treated_range[0], control_range[0])
            overlap_max = min(treated_range[1], control_range[1])
            
            if overlap_min >= overlap_max:
                print(f"WARNING: No overlap in {confounder}")
                print(f"  Treated range: {treated_range[0]:.2f} to {treated_range[1]:.2f}")
                print(f"  Control range: {control_range[0]:.2f} to {control_range[1]:.2f}")
                overlap_violations += 1
        
        if overlap_violations == 0:
            print(f"Overlap assumption:   PASSED (all confounders overlap)")
        
        # Final verdict
        print()
        if all_passed and overlap_violations == 0:
            print("Overall: ALL CHECKS PASSED - causal estimates are reliable")
        else:
            print("Overall: SOME CHECKS FAILED - interpret results carefully")
        
        return self

    def fit(self, data):
        features = [self.treatment] + self.confounders
        X = data[features].values
        y = data[self.outcome].values

        self.model = LogisticRegression()
        self.model.fit(X, y)

        X_naive = data[[self.treatment]].values
        naive_model = LogisticRegression()
        naive_model.fit(X_naive, y)
        self.results['naive_odds_ratio'] = math.exp(naive_model.coef_[0][0])

        X_conf = data[self.confounders].values
        X_treat = data[self.treatment].values
        conf_model = LinearRegression()
        conf_model.fit(X_conf, X_treat)
        r2 = r2_score(X_treat, conf_model.predict(X_conf))
        self.results['confounding_severity'] = r2

        self.results['treatment_effect_log_odds'] = self.model.coef_[0][0]
        self.results['treatment_odds_ratio'] = math.exp(self.model.coef_[0][0])
        self.results['confounder_odds_ratios'] = [
            math.exp(c) for c in self.model.coef_[0][1:]
        ]

        return self

    def ate(self, data):
        features = [self.treatment] + self.confounders

        data_treated = data[features].copy()
        data_treated[self.treatment] = 1

        data_control = data[features].copy()
        data_control[self.treatment] = 0

        prob_treated = self.model.predict_proba(
            data_treated.values
        )[:, 1]

        prob_control = self.model.predict_proba(
            data_control.values
        )[:, 1]

        individual_effects = prob_treated - prob_control
        ate_value = individual_effects.mean()

        self.results['ate'] = ate_value
        self.results['individual_effects'] = individual_effects

        return ate_value
    
    def hte(self, data):
    # We need individual effects calculated first
        if 'individual_effects' not in self.results:
            self.ate(data)
        
        individual_effects = self.results['individual_effects']
        
        # Find what predicts individual treatment effect size
        # Using confounders as predictors of response
        from sklearn.linear_model import LinearRegression
        import numpy as np
        
        X_conf = data[self.confounders].values
        
        hte_model = LinearRegression()
        hte_model.fit(X_conf, individual_effects)
        
        self.results['hte_coefficients'] = dict(
            zip(self.confounders, hte_model.coef_)
        )
        
        # Find top and bottom responders
        self.results['individual_effects'] = individual_effects
        
        # Split into high and low responders
        median_effect = np.median(individual_effects)
        high_responders = data[individual_effects > median_effect]
        low_responders = data[individual_effects <= median_effect]
        
        self.results['high_responder_avg_effect'] = individual_effects[
            individual_effects > median_effect
        ].mean()
        
        self.results['low_responder_avg_effect'] = individual_effects[
            individual_effects <= median_effect
        ].mean()
        
        self.results['high_responder_profile'] = high_responders[
            self.confounders
        ].mean().to_dict()
        
        self.results['low_responder_profile'] = low_responders[
            self.confounders
        ].mean().to_dict()
    
        return self
    def predict_effect(self, user_profile):
        """
        Predict the treatment effect for a new user
        given their confounder values.
        
        user_profile: dict like {'tech_savvy': 1.5}
        """
        if 'hte_coefficients' not in self.results:
            raise ValueError("Run hte() first")
        
        import numpy as np
        
        effect = self.results['ate']
        
        for confounder, coef in self.results['hte_coefficients'].items():
            effect += coef * user_profile.get(confounder, 0)
        
        return effect

    def summary(self):
        naive = self.results['naive_odds_ratio']
        corrected = self.results['treatment_odds_ratio']
        severity = self.results['confounding_severity'] * 100

        print(f"\n--- CausalLift Results ---")
        print(f"Naive effect (uncorrected):     {naive:.2f}x")
        print(f"True causal effect (corrected): {corrected:.2f}x")
        print(f"Confounding severity score:     {severity:.1f}%")
        print()

        if severity < 5:
            verdict = "LOW - naive measurement was mostly reliable"
        elif severity < 20:
            verdict = "MEDIUM - naive measurement was somewhat misleading"
        else:
            verdict = "HIGH - naive measurement was seriously wrong"

        print(f"Verdict: {verdict}")
        print()
        print(f"Without CausalLift you would have reported {naive:.2f}x")
        print(f"The true effect is {corrected:.2f}x")
        print(f"You were off by {abs(naive-corrected):.2f}x")

        if 'ate' in self.results:
            ate = self.results['ate']
            print(f"\nAverage Treatment Effect (ATE):  {ate:.4f}")
            print(f"In plain English: the treatment causes {ate*100:.1f} extra outcomes per 100 people")
        if 'hte_coefficients' in self.results:
            print(f"\n--- Heterogeneous Treatment Effects ---")
            print(f"What predicts who responds most to treatment:")
            for confounder, coef in self.results['hte_coefficients'].items():
                direction = "more" if coef > 0 else "less"
                print(f"  Higher {confounder} → responds {direction} (coef: {coef:.4f})")
            print()
            print(f"High responders: {self.results['high_responder_avg_effect']*100:.1f} extra outcomes per 100")
            print(f"Low responders:  {self.results['low_responder_avg_effect']*100:.1f} extra outcomes per 100")
            print()
            print(f"High responder profile:")
            for k, v in self.results['high_responder_profile'].items():
                print(f"  Average {k}: {v:.2f}")
            print(f"Low responder profile:")
            for k, v in self.results['low_responder_profile'].items():
                print(f"  Average {k}: {v:.2f}")
        return self