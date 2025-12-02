import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class LogicConstrainedExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names  # List of all OHE columns in order

    def generate_constrained_samples(self, formal_region, original_instance, n_samples=1000):
        """
        Generates synthetic samples strictly inside the formal region.
        """
        n_features = len(self.feature_names)
        samples = np.zeros((n_samples, n_features))
        
        # We iterate through every column the model expects
        for col_idx, col_name in enumerate(self.feature_names):
            
            # 1. Check if this column is part of a feature in the Formal Region
            # Logic: if col_name is 'age', it's numeric. 
            # If col_name is 'workclass_Private', base feature is 'workclass'.
            
            # Heuristic to split OHE names (assumes 'feature_value' or 'feature=value')
            if '_' in col_name:
                base_feat, cat_val = col_name.rsplit('_', 1)
            else:
                base_feat = col_name
                cat_val = None

            # 2. Apply Constraints
            if base_feat in formal_region:
                constraint = formal_region[base_feat]
                
                # Case A: Categorical Constraint (e.g., workclass == Private)
                if constraint[0] == 'eq':
                    required_val = constraint[1]
                    # If this specific OHE column matches the required value, set to 1
                    if cat_val == required_val:
                        samples[:, col_idx] = 1.0
                    else:
                        samples[:, col_idx] = 0.0
                
                # Case B: Numeric Constraint (e.g., age between 25.5 and 27.5)
                else:
                    low, high = constraint
                    samples[:, col_idx] = np.random.uniform(low, high, n_samples)
            
            else:
                # Case C: Feature NOT constrained by logic
                # STRATEGY: Fix to the original instance's value to reduce noise.
                # (We want to see what matters *given* the logical constraints)
                orig_val = original_instance[col_idx]
                samples[:, col_idx] = orig_val

        return samples

    def explain(self, formal_region, original_instance, n_samples=1000):
        # 1. Generate valid samples
        X_synth = self.generate_constrained_samples(formal_region, original_instance, n_samples)
        
        # 2. Get Model Predictions (Probabilities)
        # We predict the probability of class 1 usually
        y_synth = self.model.predict_proba(X_synth)[:, 1]
        
        # 3. Train Surrogate (Linear Regression)
        # We want to know: Within this box, how do features move the probability?
        surrogate = LinearRegression()
        surrogate.fit(X_synth, y_synth)
        
        # 4. Extract Weights
        coeffs = zip(self.feature_names, surrogate.coef_)
        # Filter out 0 weights (features that were constant/fixed)
        explanation = [(f, w) for f, w in coeffs if abs(w) > 0.0001]
        explanation.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return explanation