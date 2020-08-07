import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from mashap_lime.prepare_data import fetch_data
from mashap_lime.models import train_models
from mashap_lime.explainers import lime_explainer, mashap_explainer
from mashap_lime.comparison_metrics import timeit_context


def train_cache_models(datasets):
    """
    Train 5 models on the dataset and cache them in 'cache/' directory
    """
    trained_models_dict = dict()
    for dataset, version, task in datasets:
        print(dataset)
        x, y = fetch_data(dataset, version)
        trained_models = train_models(x, y, task)
        trained_models_dict.setdefault(dataset, trained_models)

    joblib.dump(trained_models_dict, "cache/trained_models.odict")
    return trained_models_dict


def get_random_idx(x, n):
    np.random.seed(42)
    return list(np.random.choice(x.index.to_list(), n, replace=False))


def make_test_set_idx(datasets, trained_models_dict):
    """
    Create random indices for test set, so both MASHAP and LIME can use the same ones in the
    experiments
    """
    idx_dict = dict()
    for dataset, version, mode in datasets:
        print(f"------------------- {dataset, mode} -------------------")
        x, y = fetch_data(dataset, version)
        _, x_test, _, _ = train_test_split(x, y, test_size=0.35, random_state=42)


        idx_dict_i = dict()
        for model_key, model in trained_models_dict.get(dataset).items():
            py_test = model.predict(x_test)
            if mode == "classification":
                x_test_positive = x_test[py_test == 1]
                x_test_negative = x_test[py_test == 0]
                try:
                    x_test_mix = x_test_positive.loc[
                        get_random_idx(x_test_positive, 50)
                    ].append(x_test_negative.loc[get_random_idx(x_test_negative, 50)])
                except ValueError:
                    x_test_mix = x_test[:100]
            elif mode == "regression":
                x_test_mix = x_test[:100]
            else:
                raise ValueError()
            idx_dict_i.setdefault(model_key, x_test_mix.index)
        idx_dict.setdefault(dataset, idx_dict_i)
    joblib.dump(idx_dict, "cache/idx_dict.odict")
    return idx_dict


def calculate_cache_scores(datasets, trained_models_dict, algorithm):
    """
    Calculate MASHAP and LIME scores on each dataset and each model, store result in a dictionary
    and then cache it in 'cache/'
    """
    predict_fn = None
    scores_dict = dict()
    time_dict = dict()
    idx_dict = joblib.load("cache/idx_dict.odict")

    for dataset, version, mode in datasets:
        print(f"------------------- {dataset, algorithm} -------------------")
        x, y = fetch_data(dataset, version)
        x_train, x_test, _, _ = train_test_split(x, y, test_size=0.3, random_state=42)

        scores_dict_i = dict()
        time_dict_i = dict()
        for model_key, model in trained_models_dict.get(dataset).items():
            idx = idx_dict.get(dataset).get(model_key)
            x_test_mix = x_test.loc[idx]

            if mode == "classification":
                predict_fn = model.predict_proba
            elif mode == "regression":
                predict_fn = model.predict
            else:
                raise ValueError()

            time_extractor_from_ctx_mngr = [0]

            if algorithm == "lime":
                with timeit_context(
                    f"[{model}] {algorithm} runtime:", time_extractor_from_ctx_mngr
                ):
                    scores = lime_explainer(
                        x_train, predict_fn, x_test_mix, mode=mode
                    )
            elif algorithm == "mashap":
                with timeit_context(
                    f"[{model}] {algorithm} runtime:", time_extractor_from_ctx_mngr
                ):
                    py_train = model.predict(x_train)
                    scores = mashap_explainer(x_train, py_train, x_test_mix)
            else:
                raise ValueError()
            scores_dict_i.setdefault(model_key, scores)
            time_dict_i.setdefault(model_key, time_extractor_from_ctx_mngr[0])
            print(time_dict_i)

        scores_dict.setdefault(dataset, scores_dict_i)
        time_dict.setdefault(dataset, time_dict_i)
        joblib.dump(scores_dict, f"cache/{algorithm}_scores.odict")
        joblib.dump(time_dict, f"cache/{algorithm}_runtime.odict")
    joblib.dump(scores_dict, f"cache/{algorithm}_scores.odict")
    joblib.dump(time_dict, f"cache/{algorithm}_runtime.odict")
    return (
        scores_dict,
        time_dict,
    )
