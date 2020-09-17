import numpy as np
import shap
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from xgboost import XGBRFClassifier, XGBRFRegressor
from sklearn.utils.multiclass import type_of_target
from sklearn.utils import resample
from tqdm import tqdm
import inspect


def lime_explainer(x_train, predict_fn, x_test, mode):
    num_features = x_train.shape[1]
    feature_names = x_train.columns.to_list()
    categorical_features = [
        i for i, col in enumerate(x_train.columns) if np.unique(x_train[col]).size < 10
    ]
    lime_tab_explainer = LimeTabularExplainer(
        training_data=x_train.values,
        mode=mode,
        feature_names=feature_names,
        categorical_features=categorical_features,
        random_state=42,
    )
    scores = []
    for x_i in tqdm(x_test.values):
        expl = lime_tab_explainer.explain_instance(
            data_row=x_i, predict_fn=predict_fn, num_features=num_features
        )
        scores_i = sorted(expl.local_exp.get(1))
        scores.append([value for key, value in scores_i])
    return np.array(scores)


def mashap_explainer(x_train, py_train, x_test, py_test):
    mashap = MASHAP().fit(x_train, py_train)
    shap_values = mashap.shap_values(x_test, py_test)
    return np.array(shap_values)


class MASHAP:
    def __init__(
        self,
        data=None,
        model_output="raw",
        feature_perturbation="interventional",
        **kwargs,
    ):

        self.data = data
        self.model_output = model_output
        self.feature_perturbation = feature_perturbation
        self.kwargs = kwargs

    def shap_values(
        self,
        X,
        y,
        tree_limit=None,
        approximate=False,
        check_additivity=True,
        refit=True,
    ):
        if refit:
            self.fit(X, y, clear=False)
        self._explainer = shap.TreeExplainer(
            self._surrogate,
            data=self.data,
            model_output=self.model_output,
            feature_perturbation=self.feature_perturbation,
        )
        return self._explainer.shap_values(
            X,
            y,
            tree_limit=tree_limit,
            approximate=approximate,
            check_additivity=check_additivity,
        )

    def _set_X_y(self, X, y, clear=True):
        if hasattr(self, "_X_fit") and not clear:
            X = pd.DataFrame(X)
            y = pd.Series(y)
            self._X_fit = self._X_fit.append(X, ignore_index=True)
            self._y_fit = self._y_fit.append(y, ignore_index=True)
        else:
            self._X_fit = pd.DataFrame(X)
            self._y_fit = pd.Series(y)

    def fit(self, X, y, clear=True):
        self._set_surrogate(X, y)
        self._set_X_y(X, y, clear=clear)
        self._surrogate.fit(self._X_fit, self._y_fit)
        self.fidelity_ = self._surrogate.score(self._X_fit, self._y_fit)
        return self

    def _set_surrogate(self, X, y=None):

        if not hasattr(self, "_surrogate"):
            target = type_of_target(y)
            if target == "continuous":
                self._surrogate = XGBRFRegressor(**self.kwargs)
            elif target in ["binary", "multiclass"]:
                self._surrogate = XGBRFClassifier(**self.kwargs)
            else:
                raise ValueError(
                    "Multioutput and multilabel datasets is not supported."
                )

    def __getattr__(self, name):
        explainer = self.__dict__.get("_explainer", None)
        attributes = dict(inspect.getmembers(explainer)).keys()
        if name in attributes:
            return getattr(explainer, name)
        else:
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute {name}"
            )
