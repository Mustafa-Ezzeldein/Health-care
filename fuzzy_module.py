import numpy as np
import pandas as pd
import joblib
import os

class FuzzyLogicSystem:
    def __init__(self):
        # [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, oxygen, temp]
        self.weights = np.array([0.05, 0.05, 0.2, 0.3, 0.1, 0.05, 0.05, 0.15, 0.1, 0.1, 0.05, 0.4, 0.3])
        self.weights = self.weights / np.sum(self.weights)

    def _get_feature_risk(self, idx, val):
        if idx == 2: # Chest Pain
            return val / 3.0
        if idx == 3: # BP (Stricter)
            if val >= 180: return 1.0 # Hypertensive Crisis
            if val <= 85: return 1.0  # Severe Hypotension
            if val > 140: return (val - 140) / 40
            if val < 90: return (90 - val) / 10
            return 0.0
        if idx == 4: # Cholesterol
            return min(1.0, max(0, (val - 200) / 200))
        if idx == 7: # Heart Rate
            if val > 170 or val < 50: return 1.0
            if val > 100: return (val - 100) / 70
            if val < 60: return (60 - val) / 10
            return 0.0
        if idx == 11: # SpO2
            if val < 90: return 1.0
            if val < 95: return (95 - val) / 5
            return 0.0
        if idx == 12: # Temp
            if val > 39.5 or val < 35: return 1.0
            if val > 37.5: return (val - 37.5) / 2
            return 0.0
        return 0.0

    def predict(self, features_list):
        features_array = np.array(features_list)
        if features_array.ndim == 1: features_array = features_array.reshape(1, -1)
        
        results = []
        for feat in features_array:
            risks = [self._get_feature_risk(i, feat[i]) for i in range(len(feat))]
            base_risk = np.dot(risks, self.weights) * 100
            
            # Crisis Rule: If ANY critical vital is at 1.0 (Crisis level), force high risk
            # Critical vitals indexes: 3 (BP), 11 (O2), 7 (Heart Rate)
            crisis_factors = [risks[3], risks[11], risks[7]]
            if max(crisis_factors) >= 1.0:
                final_risk = max(base_risk, 85.0) # Force at least 85% risk
            else:
                final_risk = base_risk
                
            results.append(min(100.0, final_risk))
        return np.array(results)

if __name__ == '__main__':
    fis = FuzzyLogicSystem()
    os.makedirs('models', exist_ok=True)
    joblib.dump(fis, 'models/fuzzy_model.pkl')
    print("Fuzzy Model Updated with Crisis Rules (Hypertensive/SpO2/Cardiac).")
