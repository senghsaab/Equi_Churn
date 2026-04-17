import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class EquiChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, dataset_name='saas', config=None):
        self.dataset_name = dataset_name
        self.config = config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        if self.dataset_name == 'saas':
            map_cfg = self.config['raw_mappings']
            
            # 1. Date-based transforms
            ref_date = pd.to_datetime(X[map_cfg['tenure_ref']])
            start_date = pd.to_datetime(X[map_cfg['tenure_start']])
            last_date = pd.to_datetime(X[map_cfg['last_activity']])
            
            X['tenure_months'] = (ref_date - start_date).dt.days / 30.4
            X['tenure_months'] = X['tenure_months'].clip(lower=0.1) # Avoid div by zero
            
            X['tenure_bucket'] = pd.cut(
                X['tenure_months'], 
                bins=[0, 3, 6, 12, np.inf], 
                labels=['0-3', '3-6', '6-12', '12+']
            ).astype(str)
            
            X['days_since_last_activity'] = (ref_date - last_date).dt.days
            X['near_renewal'] = (X[map_cfg['near_renewal_days']] < 30).astype(int)
            
            # 2. Usage transforms
            X['usage_intensity'] = X[map_cfg['usage_total']] / X['tenure_months']
            X['active_day_ratio'] = X[map_cfg['usage_days']] / (X['tenure_months'] * 30.4)
            X['active_day_ratio'] = X['active_day_ratio'].clip(0, 1)
            
            X['usage_trend_30d'] = (X[map_cfg['usage_l30']] - X[map_cfg['usage_p30']]) / (X[map_cfg['usage_p30']] + 1)
            
            # 3. SaaS-specific transforms
            X['seat_utilization'] = X[map_cfg['seats_active']] / X[map_cfg['seats_purchased']].clip(lower=1)
            X['feature_adoption_rate'] = X[map_cfg['features_used']] / X[map_cfg['features_avail']].clip(lower=1)
            
            # module_adoption_score (Weighted sum)
            module_cols = map_cfg['module_flags']
            weights = np.array([0.1, 0.15, 0.25, 0.2, 0.3]) # Core modules weighted higher
            X['module_adoption_score'] = (X[module_cols] * weights).sum(axis=1)
            
            X['support_ticket_rate'] = X[map_cfg['tickets_total']] / X['tenure_months']
            X['unresolved_ticket_ratio'] = X[map_cfg['tickets_open']] / X[map_cfg['tickets_total']].clip(lower=1)
            
            # 4. Revenue transforms
            X['mrr_segment'] = pd.qcut(X[map_cfg['mrr']], q=3, labels=['low', 'medium', 'high'], duplicates='drop').astype(str)
            X['revenue_per_seat'] = X[map_cfg['mrr']] / X[map_cfg['seats_purchased']].clip(lower=1)
            
        elif self.dataset_name == 'bank':
            # Simplified for Bank data
            X['tenure_months'] = X['Tenure'] * 12 # Years to months
            X['usage_intensity'] = X['IsActiveMember'] # Proxy
            
            # Safer qcut for Bank data which may have duplicate edges (e.g. many zero balances)
            try:
                X['mrr_segment'] = pd.qcut(X['Balance'], q=3, labels=['low', 'medium', 'high'], duplicates='drop').astype(str)
            except (ValueError, IndexError):
                # Fallback if too many duplicates
                X['mrr_segment'] = pd.qcut(X['Balance'].rank(method='first'), q=3, labels=['low', 'medium', 'high']).astype(str)
            
            X['near_renewal'] = 0 # Dummy for bank
            
        return X
