import numpy as np
import pandas as pd
from time import time
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score, auc, accuracy_score
from xlwt import Workbook
from math import ceil

########################################################################
# 1. Runtime
########################################################################
@contextmanager
def timeit_context(name, time_trojan):
    start_time = time()
    yield
    elapsed_time = time() - start_time
    time_trojan[0] = ceil(elapsed_time)
    print("{} {} seconds".format(name, ceil(elapsed_time)))


def runtime_calculations(mashap_d, lime_d, datasets):
    lime_time_mean, mashap_time_mean = [], []
    for ds in datasets:
        t_l = np.mean(list(lime_d.get(ds).values()))
        t_m = np.mean(list(mashap_d.get(ds).values()))
        if t_m == 0:
            t_m = 1
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
    class_idx = None
    if sign == "positive":
        ordered_features = s[s > 0].sort_values(ascending=False)
        class_idx = 1
    elif sign == "negative":
        ordered_features = s[s < 0].sort_values(ascending=True)
        class_idx = 0
    elif sign == "absolute":
        ordered_features = abs(s).sort_values(ascending=False)
    else:
        raise ValueError("'sign' valid values: 'positive', 'negative,'absolute'")
    return ordered_features, class_idx


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


def _get_prob(model, x_new, x_hide, class_idx, ordered_features, sign):
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
    if sign == "absolute":
        y_new_pred = max(model.predict_proba(np.array(x_new).reshape(1, -1)).squeeze())
    else:
        y_new_pred = model.predict_proba(np.array(x_new).reshape(1, -1)).squeeze()[
            class_idx
        ]
    y_new_pred_list.append(y_new_pred)

    # get prob. score for the first 9 fractions
    for fraction_i in range(9):
        features = ordered_features.index[
            fraction_i * fraction_size : (fraction_i * fraction_size + fraction_size)
        ]
        x_new.loc[features] = x_hide[features]
        if sign == "absolute":
            y_new_pred = max(
                model.predict_proba(np.array(x_new).reshape(1, -1)).squeeze()
            )
        else:
            y_new_pred = model.predict_proba(np.array(x_new).reshape(1, -1)).squeeze()[
                class_idx
            ]
        y_new_pred_list.append(y_new_pred)

    # get prob. score for the last fraction
    features = ordered_features.index[-last_fraction_size:]
    x_new.loc[features] = x_hide[features]
    if sign == "absolute":
        y_new_pred = max(model.predict_proba(np.array(x_new).reshape(1, -1)).squeeze())
    else:
        y_new_pred = model.predict_proba(np.array(x_new).reshape(1, -1)).squeeze()[
            class_idx
        ]
    y_new_pred_list.append(y_new_pred)

    return y_new_pred_list


def consistency_metric(x_train, x_test, y_test, metric, scores, sign, hide_mode, model):
    """
    Calculates consistency metrics: Keep/Remove positive/negative/absolute (mask/resample/impute)
    for a given test set of 50 instances. If sign is positive or absolute, test set contains
    instances with positive predictions, while if sign is negative test set contains instances with
    negative predictions (this is done so that explanations are more meaningful)
    """
    y_100_list = []

    for score, (x_idx, x_i) in zip(scores, x_test.iterrows()):
        s = pd.Series(score, index=x_test.columns)

        ordered_features, class_idx = _sort_features(sign, s)

        if metric == "keep":
            x_new = _hide_features(hide_mode, x_train, x_test)
            x_hide = x_i.copy()
            y_100_list.append(
                _get_prob(model, x_new, x_hide, class_idx, ordered_features, sign)
            )
        elif metric == "remove":
            x_new = x_i.copy()
            x_hide = _hide_features(hide_mode, x_train, x_test)
            y_100_list.append(
                _get_prob(model, x_new, x_hide, class_idx, ordered_features, sign)
            )
        else:
            raise ValueError()

    if sign == "absolute":
        return np.array(
            [roc_auc_score(y_test, np.array(y_100_list)[:, i]) for i in range(11)]
        )
    else:
        return np.mean(y_100_list, axis=0)


def write_excel(mashap_consistency_dict, lime_consistency_dict, datasets):
    wb = Workbook()
    sheet1 = wb.add_sheet("Sheet 1")
    sheet2 = wb.add_sheet("Sheet 2")

    for sheet in [sheet1, sheet2]:
        sheet.write(0, 0, "dataset")
        sheet.write(0, 1, "model")
        sheet.write(0, 2, "metric")
        sheet.write(0, 3, "mashap")
        sheet.write(0, 4, "lime")

    for metric, sheet in zip(["keep", "remove"], [sheet1, sheet2]):
        row = 0
        x = np.linspace(0, 1, 11)
        for ds in datasets:
            for model in ["knn", "dt", "rf", "gbc", "mlp"]:
                for sign in ["positive", "negative", "absolute"]:
                    for hide_m in ["mask", "resample", "impute"]:
                        row += 1
                        auc_mashap = auc(
                            x, mashap_consistency_dict[ds][model][metric][sign][hide_m]
                        )
                        auc_lime = auc(
                            x, lime_consistency_dict[ds][model][metric][sign][hide_m]
                        )
                        sheet.write(row, 0, ds)
                        sheet.write(row, 1, model)
                        sheet.write(row, 2, " ".join([metric, sign, hide_m]))
                        sheet.write(row, 3, auc_mashap)
                        sheet.write(row, 4, auc_lime)

    wb.save("comparison.xls")
