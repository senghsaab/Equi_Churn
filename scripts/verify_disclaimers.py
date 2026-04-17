"""Verify dataset source labeling and DPDP compliance for CI/CD."""
import sys

# Check 1: Data ingestion has dataset source labeling
with open('src/data/data_ingestion.py', encoding='utf-8') as f:
    content = f.read()
assert 'dataset_source' in content, 'Dataset source labeling missing from data pipeline'
assert 'kaggle_proxy' in content, 'Kaggle proxy labeling missing from data pipeline'
print('[OK] Dataset source labeling found in data_ingestion.py')

# Check 2: API has DPDP Act reference
with open('src/api.py', encoding='utf-8') as f:
    api_content = f.read()
assert 'DPDP' in api_content, 'DPDP Act reference missing from API'
print('[OK] DPDP Act 2023 reference found in api.py')

# Check 3: Fairness config validation
import yaml
with open('configs/fairness_config.yaml', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
assert 'fairness_threshold' in cfg, 'Fairness threshold missing'
assert cfg['fairness_threshold'] <= 0.15, f'Threshold {cfg["fairness_threshold"]} too permissive'
print('[OK] Fairness config validated')

print('\nAll dataset documentation checks passed')
