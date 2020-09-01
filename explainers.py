import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from xgboost import DMatrix
from xgboost import train as xgb_train
from sklearn.utils.multiclass import type_of_target
from sklearn.utils import resample
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


def mashap_explainer(x_train, py_train, x_test, py_test):
    mashap = MashapExplainer(x_train, py_train)
    mashap.partial_fit(x_test, py_test)
    shap_values = mashap.shap_values(x_test)
    return np.array(shap_values)


class MashapExplainer:

    objective_dict = {
        "continuous": "reg:squarederror",
        "binary": "binary:logistic",
        "multiclass": "multi:softmax",
    }

    def __init__(self, x, py):
        self.target_type = type_of_target(py)
        self.surrogate_model = self._create_surrogate(x, py)
        self.explainer_ = shap.TreeExplainer(model=self.surrogate_model,
                                             data=resample(x, n_samples=100, random_state=0),
                                             feature_perturbation='interventional')

    def _create_surrogate(self, x, py):
        params_1 = {
            "max_depth": 7,
            "objective": MashapExplainer.objective_dict.get(self.target_type),
            "eval_metric": "logloss",
        }
        params_2 = {"num_boost_round": 150}
        dtrain = DMatrix(x, label=py)
        booster = xgb_train(params_1, dtrain, **params_2)
        return booster

    def partial_fit(self, x_partial, y_partial):
        params_1 = {
            "max_depth": 6,
            "objective": MashapExplainer.objective_dict.get(self.target_type),
            "eval_metric": "logloss",
        }
        params_2 = {"num_boost_round": 200}
        dtrain = DMatrix(x_partial, label=y_partial)

        booster = xgb_train(
            params_1, dtrain, xgb_model=self.surrogate_model, **params_2
        )

        self.surrogate_model = booster
        self.explainer_ = shap.TreeExplainer(self.surrogate_model)

    def shap_values(self, X, y=None, tree_limit=None, approximate=False):
        """
        Calculates and returns the Shapley values through the  TreeSHAP method
        """
        return self.explainer_.shap_values(
            X, y=y, tree_limit=tree_limit, approximate=approximate
        )
