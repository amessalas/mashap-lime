import numpy as np
import pandas as pd
from time import time
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import auc
from xlwt import Workbook


########################################################################
# 1. Runtime
########################################################################
@contextmanager
def timeit_context(name, time_trojan):
    start_time = time()
    yield
    elapsed_time = time() - start_time
    time_trojan[0] = int(elapsed_time)
    print("{} {} seconds".format(name, int(elapsed_time)))


def runtime_calculations(mashap_d, lime_d):
    datasets = [
        "adult",
        "default-of-credit-card-clients",
        "musk",
        #"parkinson-speech-uci",
      #  "house_sales",
    ]
    lime_time_mean, mashap_time_mean = [], []
    for ds in datasets:
        t_l = np.mean(list(lime_d.get(ds).values()))
        t_m = np.mean(list(mashap_d.get(ds).values()))
        lime_time_mean.append(t_l)
        mashap_time_mean.append(t_m)
        print(f"[{ds}] MASHAP was {t_l/t_m: .2f} times faster")

    t1 = np.mean(lime_time_mean)
    t2 = np.mean(mashap_time_mean)
    print(f"\n\nLIME overall mean runtime {int(t1)} seconds")
    print(f"MASHAP overall mean runtime {int(t2)} seconds")
    print(f"MASHAP was approximately {t1/t2: .2f} times faster")


########################################################################
# 3. Separability
########################################################################
def check_separability(scores):
    pass


########################################################################
# 6. Consistency
########################################################################


def _calculate_sigma(x, mu):
    """
    calculates standard deviation
    """
    diffs = x - mu
    sigma = np.dot(diffs.T, diffs) / x.shape[0]
    return sigma


def _sort_features(sign, s):
    """
    orders features based on `sign`. The positive and absolute features are sorted descendingly
    while the negative features are sorted ascendingly
    """
    class_index = None
    if sign == "positive":
        ordered_features = s[s > 0].sort_values(ascending=False)
        class_index = 1
    elif sign == "negative":
        ordered_features = s[s < 0].sort_values(ascending=True)
        class_index = 0
    elif sign == "absolute":
        ordered_features = abs(s).sort_values(ascending=False)
    else:
        raise ValueError("'sign' valid values: 'positive', 'negative,'absolute'")
    return ordered_features, class_index


def _hide_features(hide_mode, x_train, x_test):
    """
    Returns an array to be used to hide features in 3 ways: 'mask' returns the mean of the features,
    'resample' returns values  from  a  random training sample, and 'impute' returns imputed values
     that match the maximum likelihood estimate under the assumption that the inputs features follow
     a multivariate normal distribution
    """
    x_new = pd.Series(None, dtype=float)
    if hide_mode == "mask":
        x_new = pd.Series(np.mean(x_test, axis=0))
    elif hide_mode == "resample":
        np.random.seed(42)
        rand_idx = np.random.choice(x_train.index.to_list(), x_train.columns.size)
        for i, col in enumerate(x_train.columns):
            x_new[col] = x_train.loc[rand_idx[i], col]
    elif hide_mode == "impute":
        mu = np.mean(x_train.values, axis=0, dtype=np.float)
        sigma = _calculate_sigma(x_train.values, mu)
        x_new = np.random.multivariate_normal(mu, sigma, size=1)
        x_new = pd.Series(x_new[0], index=x_test.columns)
    else:
        raise ValueError()
    return x_new


def _get_prob(model, x_new, x_i, class_index, ordered_features, sign):
    """
    Divides data in 11 fractions. For each fraction, it returns the probability score of the model
    on the fraction. The first fraction contains all the hidden features, the second fraction
    contains the most 1/10 positive/negative/absolute features, etc. If sign is positive/negative
    the probability score for the positive/negative class is returned, while if sign is absolute,
    the maximum probability is returned
    """
    fraction_size = len(ordered_features) // 10
    last_fraction_size = len(ordered_features) % 10

    y_new_pred_list = []

    # get prob. score for hidden values
    if sign == 'absolute':
        y_new_pred = max(model.predict_proba(np.array(x_new).reshape(1, -1)).squeeze())
    else:
        y_new_pred = model.predict_proba(np.array(x_new).reshape(1, -1)).squeeze()[class_index]
    y_new_pred_list.append(y_new_pred)

    # get prob. score for the first 9 fractions
    for fraction_i in range(9):
        features = ordered_features.index[
            fraction_i * fraction_size: (fraction_i * fraction_size + fraction_size)
        ]
        x_new.loc[features] = x_i[features]
        if sign == 'absolute':
            y_new_pred = max(model.predict_proba(np.array(x_new).reshape(1, -1)).squeeze())
        else:
            y_new_pred = model.predict_proba(np.array(x_new).reshape(1, -1)).squeeze()[class_index]
        y_new_pred_list.append(y_new_pred)

    # get prob. score for the last fraction
    features = ordered_features.index[-last_fraction_size:]
    x_new.loc[features] = x_i[features]
    if sign == 'absolute':
        y_new_pred = max(model.predict_proba(np.array(x_new).reshape(1, -1)).squeeze())
    else:
        y_new_pred = model.predict_proba(np.array(x_new).reshape(1, -1)).squeeze()[class_index]
    y_new_pred_list.append(y_new_pred)

    return y_new_pred_list


def consistency_metric(x_train, x_test, y_test, metric, scores, sign, hide_mode, model):
    """
    Calculates consistency metrics: Keep/Remove positive/negative/absolute (mask/resample/impute)
    for a given test set of 50 instances. If sign is positive or absolute, test set contains
    instances with positive predictions, while if sign is negative test set contains instances with
    negative predictions (this is done so that explanations are more meaningful)
    """
    y_50_list = []
    if sign in ["positive", "absolute"]:
        idx = range(0, 50)
    elif sign == "negative":
        idx = range(50, 100)
    else:
        raise ValueError()

    for score, (x_idx, x_i) in zip(scores, x_test.iloc[idx].iterrows()):
        s = pd.Series(score, index=x_test.columns)

        ordered_features, class_index = _sort_features(sign, s)

        if metric == 'keep':
            x_new = _hide_features(hide_mode, x_train, x_test)
            y_50_list.append(_get_prob(model, x_new, x_i, class_index, ordered_features, sign))
        elif metric == 'remove':
            x_hide = _hide_features(hide_mode, x_train, x_test)
            x_new = x_i.copy()
            y_50_list.append(_get_prob(model, x_new, x_hide, class_index, ordered_features, sign))
        else:
            raise ValueError()

    if sign == "absolute":
        return np.array(
            [
                roc_auc_score(y_test.iloc[idx], np.array(y_50_list)[:, i])
                for i in range(11)
            ]
        )
    else:
        return np.mean(y_50_list, axis=0)


def consistency_results(mashap_d, lime_d, datasets):
    """
    Calculates all consistency metrics for MASHAP and LIME scores
    """
    d_get = lambda d, ds, model, metric, sign, hide_m: d[ds][model][metric][sign][hide_m]
    x = np.linspace(0, 1, 11)
    # Overall results
    model_keys = ["knn", "dt", "rf", "gbc", "mlp"]

    results_dict_keep = dict()
    for sign in ['positive', 'negative', 'absolute']:
        results_dict_i = dict()
        for hide_mode in ['mask', 'resample', 'impute']:
            auc_mashap = []
            auc_lime = []
            for ds in datasets:
                for model in model_keys:
                    y = d_get(mashap_d, ds, model, 'keep', sign, hide_mode)
                    auc_mashap.append(auc(x, y))
                    y = d_get(lime_d, ds, model, 'keep', sign, hide_mode)
                    auc_lime.append(auc(x, y))
            results_dict_i.setdefault(hide_mode, [x > y for x, y in zip(auc_mashap, auc_lime)])
        results_dict_keep.setdefault(sign, results_dict_i)

    results_dict_remove = dict()
    for sign in ['positive', 'negative', 'absolute']:
        results_dict_i = dict()
        for hide_mode in ['mask', 'resample', 'impute']:
            auc_mashap = []
            auc_lime = []
            for ds in datasets:
                for model in model_keys:
                    y = d_get(mashap_d, ds, model, 'remove', sign, hide_mode)
                    auc_mashap.append(auc(x, y))
                    y = d_get(lime_d, ds, model, 'remove', sign, hide_mode)
                    auc_lime.append(auc(x, y))
            results_dict_i.setdefault(hide_mode, [x < y for x, y in zip(auc_mashap, auc_lime)])
        results_dict_remove.setdefault(sign, results_dict_i)

    return results_dict_keep, results_dict_remove


def write_excel(mashap_consistency_dict, lime_consistency_dict, datasets):
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1')
    sheet2 = wb.add_sheet('Sheet 2')

    for sheet in [sheet1, sheet2]:
        sheet.write(0, 1, 'dataset')
        sheet.write(0, 2, 'model')
        sheet.write(0, 3, 'metric')
        sheet.write(0, 4, 'mashap')
        sheet.write(0, 5, 'lime')

    for metric, sheet in zip(['keep', 'remove'], [sheet1, sheet2]):
        row = 0
        x = np.linspace(0, 1, 11)
        for ds in datasets:
            for model in ["knn", "dt", "rf", "gbc", "mlp"]:
                for sign in ['positive', 'negative', 'absolute']:
                    for hide_m in ['mask', 'resample', 'impute']:
                        row += 1
                        auc_mashap = auc(x, mashap_consistency_dict[ds][model][metric][sign][hide_m])
                        auc_lime = auc(x, lime_consistency_dict[ds][model][metric][sign][hide_m])
                        sheet.write(row, 1, ds)
                        sheet.write(row, 2, model)
                        sheet.write(row, 3, " ".join([metric, sign, hide_m]))
                        sheet.write(row, 4, auc_mashap)
                        sheet.write(row, 5, auc_lime)

    wb.save('comparison.xls')