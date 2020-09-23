import joblib
import os
from cache import (
    train_cache_models,
    calculate_cache_scores,
    make_test_set_idx,
)
from comparison_metrics import (
    get_consistency_metrics,
    runtime_calculations,
    write_excel,
    t_test
)


########################################################################
# define datasets
########################################################################
os.makedirs("cache", exist_ok=True)
openml_datasets_ids = [
    ("adult", 2, "classification"),
    ("default-of-credit-card-clients", "active", "classification"),
    ("musk", "active", "classification"),
    ("hill-valley", 1, "classification"),
    ("ozone-level-8hr", "active", "classification"),
    ("pc1", "active", "classification"),
    ("compas-two-years", 3, "classification"),
    ("elevators", 2, "classification"),
    ("spambase", "active", "classification"),
    ("climate-model-simulation-crashes", 1, "classification"),
    ("kr-vs-kp", "active", "classification"),
    ("cylinder-bands", "active", "classification"),
    ("ionosphere", "active", "classification"),
    ("kc3", "active", "classification"),
    ("qsar-biodeg", "active", "classification"),
    ("SPECTF", 1, "classification"),
    ("credit-g", "active", "classification"),
    ("kc1", "active", "classification"),
    ("mushroom", "active", "classification"),
    ("ringnorm", "active", "classification"),
    ("twonorm", "active", "classification"),
    ("bank-marketing", 1, "classification"),
    ("vote", 1, "classification"),
    ("credit-approval", "active", "classification"),
]
########################################################################
# train models on every dataset (and cache them)
########################################################################
try:
    trained_models_dict = joblib.load("cache/trained_models.dict")
except FileNotFoundError:
    print("========== TRAINING MODELS ==========")
    trained_models_dict = train_cache_models(openml_datasets_ids)

########################################################################
# measure MASHAP and LIME scores (and cache them)
########################################################################
try:
    idx_dict = joblib.load(f"cache/idx_dict.dict")
except FileNotFoundError:
    print("========== MAKING FIXED TEST SETS ==========")
    idx_dict = make_test_set_idx(openml_datasets_ids, trained_models_dict)

try:
    mashap_scores_dict = joblib.load(f"cache/mashap_scores.dict")
    mashap_runtime_dict = joblib.load(f"cache/mashap_runtime.dict")
except FileNotFoundError:
    print("========== CALCULATING MASHAP SCORES ==========")
    (mashap_scores_dict, mashap_runtime_dict,) = calculate_cache_scores(
        openml_datasets_ids, trained_models_dict, idx_dict, "mashap"
    )
    joblib.dump(mashap_scores_dict, "cache/mashap_scores.dict")
    joblib.dump(mashap_runtime_dict, "cache/mashap_runtime.dict")


try:
    lime_scores_dict = joblib.load(f"cache/lime_scores.dict")
    lime_runtime_dict = joblib.load(f"cache/lime_runtime2.dict")
except FileNotFoundError:
    print("========== CALCULATING LIME SCORES ==========")
    (lime_scores_dict, lime_runtime_dict,) = calculate_cache_scores(
        openml_datasets_ids, trained_models_dict, idx_dict, "lime"
    )

########################################################################
# measure MASHAP and LIME consistency metrics (and cache them)
########################################################################

try:
    mashap_consistency_dict = joblib.load("cache/mashap_consistency.dict")
except FileNotFoundError:
    print("========== CALCULATING MASHAP CONSISTENCY METRICS ==========")
    mashap_scores_dict = joblib.load(f"cache/mashap_scores.dict")
    mashap_consistency_dict = get_consistency_metrics(
        openml_datasets_ids, algorithm="mashap", scores_dict=mashap_scores_dict, idx_dict=idx_dict, trained_models_dict=trained_models_dict
    )
    joblib.dump(mashap_consistency_dict, "cache/mashap_consistency.dict")

try:
    lime_consistency_dict = joblib.load("cache/lime_consistency.dict")
except FileNotFoundError:
    print("========== CALCULATING LIME CONSISTENCY METRICS ==========")
    lime_consistency_dict = get_consistency_metrics(
        openml_datasets_ids, algorithm="lime", scores_dict=lime_scores_dict, idx_dict=idx_dict, trained_models_dict=trained_models_dict
    )
    joblib.dump(lime_consistency_dict, "cache/lime_consistency.dict")


########################################################################
# print/save results
########################################################################
datasets = [ds for ds, v, t in openml_datasets_ids]
os.makedirs("results", exist_ok=True)

runtime_calculations(mashap_runtime_dict, lime_runtime_dict, datasets)
write_excel(mashap_consistency_dict, lime_consistency_dict, datasets, 'results/comparison_mashap_lime')
print(t_test('results/comparison_mashap_lime.xls'))
with open('results/t_test_results.txt', 'w') as f:
    f.write(t_test('results/comparison_mashap_lime.xls').to_string())
