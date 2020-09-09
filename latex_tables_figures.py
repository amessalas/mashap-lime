import numpy as np
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from prepare_data import fetch_data


openml_datasets = [
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


def latex_table_1(datasets):
    print('\n')
    df = pd.DataFrame([], columns=['dataset', 'records', 'features'])
    for dataset, version, mode in datasets:
        x, y = fetch_data(dataset, version)
        df.loc[df.size] = ([dataset, x.shape[0], x.shape[1]])

    df.sort_values('dataset', inplace=True)
    for i, row in df.iterrows():
        print(f"{row['dataset']} &  {row['records']} &  {row['features']}\\\\")


def figure_3(dataset, model, metric):
    mashap = joblib.load("cache/mashap_consistency.dict")
    lime = joblib.load("cache/lime_consistency.dict")
    x = [i/100 for i in range(0, 110, 10)]
    y = mashap[dataset][model][metric]['positive']['mask']
    y2 = lime[dataset][model][metric]['positive']['mask']
    ax = sns.lineplot(x, y,)
    ax = sns.lineplot(x, y2)
    ax.set_title(f'{metric.title()} positive mask metric', fontdict={'fontsize': 15})
    ax.set_xlabel('Fraction of features kept', fontsize=15)
    ax.set_ylabel("Mean model output", fontsize=15)
    plt.legend(labels=['MASHAP', 'LIME'], fontsize=12)
    plt.show()
    fig = ax.get_figure()
    return fig


def figure_4(datasets, lime_dict, mashap_dict):
    print('\n')
    df = pd.DataFrame([], columns=['dataset', 'ratio'])
    for dataset, version, mode in datasets:
        t_l = np.mean(list(lime_dict[dataset].values()))
        t_m = np.mean(list(mashap_dict[dataset].values()))
        if t_m == 0:
            t_m = 1
        ratio = t_l/t_m
        name = f"{dataset}"
        df.loc[len(df)] = ([name, ratio])
    df.sort_values('ratio', inplace=True)

    ax = sns.barplot(x='dataset', y='ratio', data=df, color='blue')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=18, horizontalalignment='right')
    for i, r in enumerate(df.ratio):
        ax.text(i, r, int(r), color='black', ha="center")

    ax.set_xlabel('Dataset', fontsize=18)
    ax.set_ylabel("Runtime Ratio of LIME over MASHAP", fontsize=18)
    sns.set_style("whitegrid")
    fig = ax.get_figure()
    return fig


latex_table_1(openml_datasets)

dataset, model, metric = "bank-marketing", 'gbc', 'keep'
fig3a = figure_3(dataset, model, metric)
fig3a.savefig(f'figures/figure3_{metric}.eps', dpi=400)
# fig3a.savefig(f'figures/figure3_{metric}.png', dpi=400)

metric = 'remove'
fig3b = figure_3(dataset, model, metric)
fig3b.savefig(f'figures/figure3_{metric}.eps', dpi=400)
# fig3b.savefig(f'figures/figure3_{metric}.png', dpi=400)


lime_runtime_dict = joblib.load('cache/lime_runtime.dict')
mashap_runtime_dict = joblib.load('cache/mashap_runtime.dict')
fig4 = figure_4(openml_datasets, lime_runtime_dict, mashap_runtime_dict)
fig4.savefig('runtime.eps', dpi=400)
# fig4.savefig('runtime.png', dpi=400)
