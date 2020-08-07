import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from shap import TreeExplainer
from shap.explainers.explainer import Explainer
from xgboost import DMatrix
from xgboost import train as xgb_train
from sklearn.utils.multiclass import type_of_target
from tqdm import tqdm


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


def mashap_explainer(x_train, py_train, x_test):
    mashap = MashapExplainer(x_train, py_train)
    shap_values = mashap.shap_values(x_test)
    return np.array(shap_values)


class MashapExplainer(Explainer):
    def __init__(self, x, py):
        surrogate_model = self._create_surrogate(x, py)
        self.explainer_ = TreeExplainer(surrogate_model)

    @staticmethod
    def _create_surrogate(x, py):
        target_type = type_of_target(py)
        objective_dict = {
            "continuous": "reg:squarederror",
            "binary": "binary:logistic",
            "multiclass": "multi:softmax",
        }
        params_1 = {
            "max_depth": 8,
            "objective": objective_dict.get(target_type),
            "eval_metric": "logloss",
        }
        params_2 = {"num_boost_round": 200}
        dtrain = DMatrix(x, label=py)
        booster = xgb_train(params_1, dtrain, **params_2)

        # monkey patch to fix xgboost error with shap 0.35.0 (https://github.com/slundberg/shap/issues/1215#issuecomment-641102855)
        model_bytearray = booster.save_raw()[4:]

        def f(self=None):
            return model_bytearray

        booster.save_raw = f
        return booster

    def shap_values(self, X, y=None, tree_limit=None, approximate=False):
        """
        Calculates and returns the Shapley values through the  TreeSHAP method
        """
        return self.explainer_.shap_values(
            X, y=y, tree_limit=tree_limit, approximate=approximate
        )
