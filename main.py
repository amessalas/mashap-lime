import joblib
import os
from tqdm import tqdm
from mashap_lime.cache import (
    train_cache_models,
    calculate_cache_scores,
    make_test_set_idx,
)
from mashap_lime.prepare_data import fetch_data
from mashap_lime.comparison_metrics import consistency_metric, runtime_calculations, \
    consistency_results, write_excel
from sklearn.model_selection import train_test_split


########################################################################
# define datasets
########################################################################
os.makedirs('cache', exist_ok=True)
openml_datasets_ids = [
    # ("adult", 2, "classification"),
    # ("default-of-credit-card-clients", "active", "classification"),
    # ("musk", "active", "classification"),
    # ("hill-valley", 1, "classification"),
    ("ozone-level-8hr", "active", "classification"),
    ("pc1", "active", "classification"),
    ("pc2", "active", "classification"),
    ("pc3", "active", "classification"),
    ("pc4", "active", "classification"),
    ("spambase", "active", "classification"),
    ("climate-model-simulation-crashes", 1, "classification"),
    ("kr-vs-kp", "active", "classification"),
    ("cylinder-bands", "active", "classification"),
    # ("ionosphere", "active", "classification"),
    # ("kc3", "active", "classification"),
    # ("qsar-biodeg", "active", "classification"),
    # ("SPECTF", 1, "classification"),
    # ("credit-g", "active", "classification"),
    # ("kc1", "active", "classification"),
    # ("mushroom", "active", "classification"),
    # ("ringnorm", "active", "classification"),
    # ("twonorm", "active", "classification"),
    # ("bank-marketing", 1, "classification"),
    # ("vote", 1, "classification"),
    # ("credit-approval", "active", "classification"),
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
# measure MASHAP and LIME scores (and cache them)import numpy as np
########################################################################
try:
    idx_dict = joblib.load(f"cache/idx_dict.dict")
except FileNotFoundError:
    idx_dict = make_test_set_idx(openml_datasets_ids, trained_models_dict)

try:
    mashap_scores_dict = joblib.load(f"cache/mashap_scores.dict")
    mashap_runtime_dict = joblib.load(f"cache/mashap_runtime.dict")
except FileNotFoundError:
    print("========== CALCULATING MASHAP SCORES ==========")
    (mashap_scores_dict, mashap_runtime_dict,) = calculate_cache_scores(
        openml_datasets_ids, trained_models_dict, "mashap"
    )

try:
    lime_scores_dict = joblib.load(f"cache/lime_scores.dict")
    lime_runtime_dict = joblib.load(f"cache/lime_runtime.dict")
except FileNotFoundError:
    print("========== CALCULATING LIME SCORES ==========")
    (lime_scores_dict, lime_runtime_dict,) = calculate_cache_scores(
        openml_datasets_ids, trained_models_dict, "lime"
    )


def get_consistency_metrics(datasets, algorithm):
    consistency_scores_dict = dict()
    model_keys = ["knn", "dt", "rf", "gbc", "mlp"]
    for dataset, version, mode in datasets:
        x, y = fetch_data(dataset, version)
        x_train, _, y_train, _ = train_test_split(x, y, test_size=0.3, random_state=42)
        model_dict = dict()

        for model_key in tqdm(model_keys):
            test_idx = idx_dict.get(dataset).get(model_key)
            x_test = x.loc[test_idx]
            y_test = y.loc[test_idx]
            scores = eval(algorithm + "_scores_dict").get(dataset).get(model_key)
            model = trained_models_dict.get(dataset).get(model_key)
            sign_dict = dict()
            metric_dict = dict()
            for metric in ["keep", "remove"]:
                for sign in ["positive", "negative", "absolute"]:
                    hide_mode_dict = dict()
                    for hide_mode in ["mask", "resample", "impute"]:
                        mean_score = consistency_metric(
                            x_train,
                            x_test.copy(),
                            y_test.copy(),
                            metric,
                            scores,
                            sign,
                            hide_mode,
                            model
                        )

                        hide_mode_dict.setdefault(hide_mode, mean_score)
                    sign_dict.setdefault(sign, hide_mode_dict)
                metric_dict.setdefault(metric, sign_dict)
            model_dict.setdefault(model_key, metric_dict)
        consistency_scores_dict.setdefault(dataset, model_dict)
    return consistency_scores_dict


try:
    mashap_consistency_dict = joblib.load('cache/mashap_consistency.dict')
    lime_consistency_dict = joblib.load('cache/lime_consistency.dict')
except FileNotFoundError:
    print("========== CALCULATING CONSISTENCY SCORES ==========")
    mashap_consistency_dict = get_consistency_metrics(openml_datasets_ids, algorithm="mashap")
    lime_consistency_dict = get_consistency_metrics(openml_datasets_ids, algorithm="lime")
    joblib.dump(mashap_consistency_dict, 'cache/mashap_consistency.dict')
    joblib.dump(lime_consistency_dict, 'cache/lime_consistency.dict')


########################################################################
# print results
########################################################################
runtime_calculations(mashap_runtime_dict, lime_runtime_dict)

datasets = [ds for ds, v, t in openml_datasets_ids]
write_excel(mashap_consistency_dict, lime_consistency_dict, datasets)


# results_dict_keep, results_dict_remove = consistency_results(mashap_consistency_dict,
#                                                              lime_consistency_dict,
#                                                              datasets)
# results = dict()
# results.setdefault('keep', results_dict_keep)
# results.setdefault('remove', results_dict_remove)
