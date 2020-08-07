import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml


def fetch_data(name, version="active", task="classification"):
    """
    Downloads data from OpenML and returns it in a uniform X, y  clean form
    """
    bunch = fetch_openml(name=name, version=version, as_frame=True)
    df = bunch.data.copy()
    cat_columns = df.select_dtypes(["category"]).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    target_name = bunch.target_names[0]
    df[target_name] = bunch.target

    if "date" in df.columns.to_list():
        df.drop(columns=["date"], inplace=True)

    if df.isna().any().any():
        df.dropna(inplace=True)

    if task == "classification":
        df[target_name] = (
            LabelEncoder().fit(np.unique(df[target_name])).transform(df[target_name])
        )
        return df[df.columns[:-1]], df[target_name]
    elif task == "regression":
        return df[df.columns[:-1]], df[target_name]
    else:
        raise ValueError("task can either be 'classification' or 'regression'.")
