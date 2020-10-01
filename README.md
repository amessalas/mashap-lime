# MASHAP: a faster alternative to LIME for model-agnostic machine learning interpretability

This repository contains all the code used for the paper: "*MASHAP: a faster alternative to LIME for model-agnostic machine learning interpretability*"

To replicate the experiments:
* Install packages in `requirements.txt` in virtual enviroment running Python 3.7
* Run `main.py`
* LIME and MASHAP scores, runtime and consistency metric measurements will be cached in a new created folder called `cache`
* Results (runtime, consistency measurements and t_test evaluation) will be stored in a new created folder called `results`

<br>
<i>Note 1: LIME calculations take a lot of time</i> <br>
<i>Note 2: additional packages (`jupyter`, `interpret`, `seaborn`) required for `lime_graph.ipynb` and `latex_tables_figures.py` which are not related to the experiments </i>
