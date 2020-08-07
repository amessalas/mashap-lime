from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split


def train_models(x, y, task):
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.3, random_state=42)
    classification_models = dict(
        {
            "knn": KNeighborsClassifier(n_neighbors=5),
            "dt": DecisionTreeClassifier(max_depth=5),
            "rf": RandomForestClassifier(max_depth=5, n_estimators=50),
            "gbc": GradientBoostingClassifier(max_depth=5, n_estimators=50),
            "mlp": MLPClassifier(random_state=42, hidden_layer_sizes=(100, 5),),
        }
    )

    regression_models = dict(
        {
            "knn": KNeighborsRegressor(n_neighbors=5),
            "dt": DecisionTreeRegressor(max_depth=5),
            "rf": RandomForestRegressor(max_depth=5, n_estimators=50),
            "gbc": GradientBoostingRegressor(max_depth=5, n_estimators=50),
            "mlp": MLPRegressor(
                random_state=42, alpha=0.5, hidden_layer_sizes=(100, 5),
            ),
        }
    )
    trained_models = dict()
    if task == "classification":
        for key, model in tqdm(classification_models.items(), position=0, leave=True):
            trained_models.setdefault(key, model.fit(x_train, y_train))
    elif task == "regression":
        for key, model in tqdm(regression_models.items(), position=0, leave=True):
            trained_models.setdefault(key, model.fit(x_train, y_train))
    else:
        raise ValueError("task can either be 'classification' or 'regression'.")

    return trained_models
