import argparse
import os
import pickle
import sys

import numpy as np
import scipy.sparse as sp
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from common.tools import *

parser = argparse.ArgumentParser()
parser.add_argument("GRAPH_VER", help="version of the graph you want regex to label your CSV with", type=str)
parser.add_argument("DATASET_PATH", help="path to your input CSV", type=str)
parser.add_argument("DATASET_PSEUDO_PATH", help="path to your input CSV", type=str)
parser.add_argument("N_TRIALS", help="optuna n trials, if 0 use default hyperparams", type=int)
args = parser.parse_args()

GRAPH_VER = args.GRAPH_VER
DATASET_PATH = args.DATASET_PATH
N_TRIALS = args.N_TRIALS
DATASET_PSEUDO_PATH = args.DATASET_PSEUDO_PATH
TAGS_TO_PREDICT = get_graph_vertices(GRAPH_VER)

MODEL_DIR = "../models/hyper_svm_regex_graph_v{}.sav".format(GRAPH_VER)
TFIDF_DIR = "../models/tfidf_hyper_svm_graph_v{}.pickle".format(GRAPH_VER)

EXPERIMENT_DATA_PATH = ".."
CODE_COLUMN = "code_block"
TARGET_COLUMN = "graph_vertex_id"

RANDOM_STATE = 42
MAX_ITER = 10000

HYPERPARAM_SPACE = {
    "svm_c": (1e-1, 1e3),
    "tfidf_min_df": (2, 6),
    "tfidf_max_df": (0.2, 1.0),
    "svm_kernel": ["linear", "poly", "rbf"],
    "svm_degree": (2, 6),  # in case of poly kernel
}

DEFAULT_HYPERPARAMS = {
    "svm__C": 145.56,
    "tfidf__min_df": 2,
    "tfidf__max_df": 0.26,
    "svm__kernel": "linear",
    "tfidf__smooth_idf": True,
    "svm__random_state": RANDOM_STATE,
    "svm__max_iter": MAX_ITER,
}


def cross_val_scores(kf, clf, X, y, pseudo_X, pseudo_y):
    f1s = []
    accuracies = []
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train = sp.vstack((X_train, pseudo_X))
        y_train = np.append(y_train, pseudo_y)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        f1s.append(f1_score(y_test, y_pred, average="weighted"))
        accuracies.append(accuracy_score(y_test, y_pred))

    f1s = np.array(f1s)
    accuracies = np.array(accuracies)
    return f1s.mean(), f1s.std(), accuracies.mean(), accuracies.std()


class Objective:
    def __init__(self, pseudo_df, df, kfold_params, svm_c, tfidf_min_df, tfidf_max_df, svm_kernel, svm_degree, use_proba=False):
        self.kf = StratifiedKFold(**kfold_params)
        self.c_range = svm_c
        self.min_df_range = tfidf_min_df
        self.max_df_range = tfidf_max_df
        self.kernels = svm_kernel
        self.poly_degrees = svm_degree
        self.df = df
        self.pseudo_df = pseudo_df
        self.use_proba = use_proba

    def __call__(self, trial):
        tfidf_params = {
            "min_df": trial.suggest_int("tfidf__min_df", *self.min_df_range),
            "max_df": trial.suggest_loguniform("tfidf__max_df", *self.max_df_range),
            "smooth_idf": True,
        }
        code_blocks_tfidf = tfidf_fit_transform(self.df[CODE_COLUMN], tfidf_params, TFIDF_DIR)
        code_blocks_tfidf_pseudo = tfidf_transform(self.pseudo_df[CODE_COLUMN], tfidf_params, TFIDF_DIR)
        X, y = code_blocks_tfidf, self.df[TARGET_COLUMN].values
        pseudo_X, pseudo_y = code_blocks_tfidf_pseudo, self.pseudo_df[TARGET_COLUMN].values

        svm_params = {
            "C": trial.suggest_loguniform("svm__C", *self.c_range),
            "kernel": trial.suggest_categorical("svm__kernel", self.kernels),
            "random_state": RANDOM_STATE,
            "max_iter": MAX_ITER,
            "probability": self.use_proba,
        }
        if svm_params["kernel"] == "poly":
            svm_params["degree"] = trial.suggest_int("svm__degree", *self.poly_degrees)
        clf = SVC(**svm_params)

        f1_mean, _, _, _ = cross_val_scores(self.kf, clf, X, y, pseudo_X, pseudo_y)
        return f1_mean


def select_hyperparams(pseudo_df, df, kfold_params, tfidf_path, model_path, use_proba=False):
    """
    Uses optuna to find hyperparams that maximize F1 score
    :param df: labelled dataset
    :param kfold_params: parameters for sklearn's KFold
    :param tfidf_dir: where to save trained tf-idf
    :return: dict with parameters and metrics
    """

    study = optuna.create_study(direction="maximize", study_name="svm with kernels")
    objective = Objective(pseudo_df, df, kfold_params, **HYPERPARAM_SPACE, use_proba=use_proba)

    if N_TRIALS > 0:
        study.optimize(objective, n_trials=N_TRIALS)
        params = study.best_params
    else:
        params = DEFAULT_HYPERPARAMS

    best_tfidf_params = {
        "smooth_idf": True,
    }
    best_svm_params = {
        "random_state": RANDOM_STATE,
        "max_iter": MAX_ITER,
    }
    for key, value in params.items():
        model_name, param_name = key.split("__")
        if model_name == "tfidf":
            best_tfidf_params[param_name] = value
        elif model_name == "svm":
            best_svm_params[param_name] = value

    code_blocks_tfidf = tfidf_fit_transform(df[CODE_COLUMN], best_tfidf_params, tfidf_path)
    X, y = code_blocks_tfidf, df[TARGET_COLUMN].values
    code_blocks_tfidf_pseudo = tfidf_transform(pseudo_df[CODE_COLUMN], best_tfidf_params, tfidf_path)
    pseudo_X, pseudo_y = code_blocks_tfidf_pseudo, pseudo_df[TARGET_COLUMN].values
    clf = SVC(**best_svm_params)

    f1_mean, f1_std, accuracy_mean, accuracy_std = cross_val_scores(objective.kf, clf, X, y, pseudo_X, pseudo_y)

    clf.fit(X, y)
    pickle.dump(clf, open(model_path, "wb"))

    metrics = dict(
        test_f1_score=f1_mean,
        test_accuracy=accuracy_mean,
        test_f1_std=f1_std,
        test_accuracy_std=accuracy_std,
    )

    return best_tfidf_params, best_svm_params, metrics


if __name__ == "__main__":
    df = load_data(DATASET_PATH)
    df_pseudo = load_data(DATASET_PSEUDO_PATH)

    print(df.columns)
    nrows = df.shape[0]
    print("loaded")

    kfold_params = {
        "n_splits": 10,
        "random_state": RANDOM_STATE,
        "shuffle": True,
    }
    data_meta = {
        "DATASET_PATH": DATASET_PATH,
        "nrows": nrows,
        "label": TAGS_TO_PREDICT,
        "model": MODEL_DIR,
        "script_dir": __file__,
    }
    print("selecting hyperparameters")
    tfidf_params, svm_params, metrics = select_hyperparams(df, df_pseudo, kfold_params, TFIDF_DIR, MODEL_DIR)
    print("hyperparams:", "\ntfidf", tfidf_params, "\nmodel", svm_params)
    print("metrics:", metrics)
    print("finished")
