import argparse
import logging
import sys

import optuna
from sklearn.model_selection import StratifiedKFold,  KFold
from sklearn.svm import SVC

from common.tools import *

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

parser = argparse.ArgumentParser()
parser.add_argument("GRAPH_VER", help="path to actual_graph CSV", type=str) # ../data_updated/actual_graph_2021-06-09.csv
parser.add_argument("DATASET_PATH", help="path to your input CSV", type=str)
parser.add_argument("N_TRIALS", help="optuna n trials, if 0 use default hyperparams", type=int)

args = parser.parse_args()

GRAPH_VER = args.GRAPH_VER
DATASET_PATH = args.DATASET_PATH
N_TRIALS = args.N_TRIALS

MODEL_DIR = "../models/hyper_svm_regex_graph_upper_clf.sav"
TFIDF_DIR = "../models/tfidf_hyper_svm_graph_upper.pickle"

RANDOM_STATE = 42
MAX_ITER = 10000

HYPERPARAM_SPACE = {
    "svm_c": (1e-1, 1e3),
    "tfidf_min_df": (1, 10),
    "tfidf_max_df": (0.2, 0.7),
    "svm_kernel": ["linear", "poly", "rbf"],
    "svm_degree": (2, 6),  # in case of poly kernel
}

DEFAULT_HYPERPARAMS_UPPER = {
    "svm__C": 149.65,
    "tfidf__min_df": 1,
    "tfidf__max_df": 0.99,
    "svm__kernel": "poly",
    "svm__degree": 2,
    "tfidf__smooth_idf": True,
    "svm__random_state": RANDOM_STATE,
    "svm__max_iter": MAX_ITER,
}


def get_ver_to_sub(graph_df):
    ver_to_sub = dict()
    for i in graph_df.index:
        ver_to_sub[i] = graph_df.graph_vertex[i]
    return ver_to_sub


def get_vertices(df, ver_to_sub):
    return df["graph_vertex_id"].apply(lambda x: ver_to_sub[x])


def cross_val_predict(kf, clf, X, y, predict=True):
    f1s = []
    accuracies = []
    if predict:
        preds = pd.DataFrame(-1, index=list(range(X.shape[0])), columns=["pred"])

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if len(set(y_train)) == 1:
            y_pred = np.array([y_train[0]] * X_train.shape[0])
        else:
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
        if predict:
            preds.loc[test_index, 'pred'] = y_pred
        f1s.append(f1_score(y_test, y_pred, average="weighted"))
        accuracies.append(accuracy_score(y_test, y_pred))

    f1s = np.array(f1s)
    accuracies = np.array(accuracies)
    if not predict:
        preds = None
    return f1s.mean(), f1s.std(), accuracies.mean(), accuracies.std(), preds


class Objective:
    def __init__(
        self,
        df,
        kfold_params,
        code_col,
        target_col,
        svm_c,
        tfidf_min_df,
        tfidf_max_df,
        svm_kernel,
        svm_degree,
    ):
        self.kf = StratifiedKFold(**kfold_params)
        self.c_range = svm_c
        self.min_df_range = tfidf_min_df
        self.max_df_range = tfidf_max_df
        self.kernels = svm_kernel
        self.poly_degrees = svm_degree
        self.df = df
        self.code_col = code_col
        self.target_col = target_col

    def __call__(self, trial):
        tfidf_params = {
            "min_df": trial.suggest_int("tfidf__min_df", *self.min_df_range),
            "max_df": trial.suggest_loguniform("tfidf__max_df", *self.max_df_range),
            "smooth_idf": True,
        }
        code_blocks_tfidf = tfidf_fit_transform(self.df[self.code_col], tfidf_params)
        X, y = code_blocks_tfidf, self.df[self.target_col].values

        svm_params = {
            "C": trial.suggest_loguniform("svm__C", *self.c_range),
            "kernel": trial.suggest_categorical("svm__kernel", self.kernels),
            "random_state": RANDOM_STATE,
            "max_iter": MAX_ITER,
        }
        if svm_params["kernel"] == "poly":
            svm_params["degree"] = trial.suggest_int("svm__degree", *self.poly_degrees)

        clf = SVC(**svm_params)

        f1_mean, _, _, _, _ = cross_val_predict(self.kf, clf, X, y, False)
        return f1_mean


def select_hyperparams(
    df,
    kfold_params,
    tfidf_path,
    model_path,
    code_col,
    target_col,
    n_trials,
    hyperparam_space,
    default_hyperparams,
):
    """
    Uses optuna to find hyperparams that maximize F1 score
    :param df: labelled dataset
    :param kfold_params: parameters for sklearn's KFold
    :param tfidf_path: where to save trained tf-idf
    :return: dict with parameters and metrics
    """

    study = optuna.create_study(direction="maximize", study_name="svm with kernels")
    objective = Objective(df, kfold_params, code_col, target_col, **hyperparam_space)

    if n_trials > 0:
        study.optimize(objective, n_trials=n_trials)
        params = study.best_params
    else:
        params = default_hyperparams

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

    code_blocks_tfidf = tfidf_fit_transform(df[code_col], best_tfidf_params, tfidf_path)
    X, y = code_blocks_tfidf, df[target_col].values
    clf = SVC(**best_svm_params)

    f1_mean, f1_std, accuracy_mean, accuracy_std, preds = cross_val_predict(
        objective.kf, clf, X, y
    )
    if model_path is not None:
        clf.fit(X, y)
        pickle.dump(clf, open(model_path, "wb"))

    metrics = dict(
        test_f1_score=f1_mean,
        test_accuracy=accuracy_mean,
        test_f1_std=f1_std,
        test_accuracy_std=accuracy_std,
    )

    return best_tfidf_params, best_svm_params, metrics, preds


if __name__ == "__main__":
    graph_df = pd.read_csv(GRAPH_VER, index_col=0)

    ver_to_sub = get_ver_to_sub(graph_df)

    df = load_data(DATASET_PATH, sep=";")
    codes, uniques = pd.factorize(get_vertices(df, ver_to_sub))
    df["graph_upper_vertex"] = codes

    print(df.columns)
    nrows = df.shape[0]
    print("loaded")

    kfold_params = {
        "n_splits": 10,
        "random_state": RANDOM_STATE,
        "shuffle": True,
    }

    print("selecting hyperparameters")
    tfidf_params, svm_params, metrics, preds_upper = select_hyperparams(
        df.drop(columns=["graph_vertex_id"]),
        kfold_params,
        TFIDF_DIR,
        MODEL_DIR,
        "code_block",
        "graph_upper_vertex",
        N_TRIALS,
        HYPERPARAM_SPACE,
        DEFAULT_HYPERPARAMS_UPPER,
    )
    print(
        "upper classifier hyperparams:", "\ntfidf", tfidf_params, "\nmodel", svm_params
    )
    print("metrics:", metrics)

    lower_vertex_params = dict()

    kfold_params = {
        "n_splits": 2,
        "random_state": RANDOM_STATE,
        "shuffle": True,
    }

    for vertex in range(len(uniques)):
        print("search params for", uniques[vertex])
        tfidf_params, svm_params, metrics, preds = select_hyperparams(
            df[df['graph_upper_vertex'] == vertex].reset_index().drop(columns=["graph_upper_vertex"]),
            kfold_params,
            TFIDF_DIR,
            None,
            "code_block",
            "graph_vertex_id",
            N_TRIALS,
            HYPERPARAM_SPACE,
            DEFAULT_HYPERPARAMS_UPPER,
        )

        lower_vertex_params[vertex] = (uniques[vertex], tfidf_params, svm_params)

        print(
            "lower classifier hyperparams for", uniques[vertex], ":", "\ntfidf", tfidf_params, "\nmodel", svm_params
        )
    # df['upper_pred'] = -1
    df['upper_pred'] = preds_upper['pred']

    final_preds = pd.DataFrame(-1, index=list(range(len(df))), columns=["pred"])
    kf = KFold(**kfold_params)

    for vertex in range(len(uniques)):
        idx = df[df['upper_pred'] == vertex].index
        df_sub = df.loc[idx].copy()
        df_sub = df_sub.drop(columns=["graph_upper_vertex", 'upper_pred'])

        _, best_tfidf_params, best_svm_params = lower_vertex_params[vertex]
        try:
            code_blocks_tfidf = tfidf_fit_transform(df_sub["code_block"], best_tfidf_params, TFIDF_DIR)
        except ValueError:
            # for min_df > max_df
            best_tfidf_params["min_df"] = 1
            best_tfidf_params["max_df"] = 0.99
            code_blocks_tfidf = tfidf_fit_transform(df_sub["code_block"], best_tfidf_params, TFIDF_DIR)

        X, y = code_blocks_tfidf, df_sub["graph_vertex_id"].values
        clf = SVC(**best_svm_params)

        _, _, _, _, preds_sub = cross_val_predict(
            kf, clf, X, y
        )
        final_preds.loc[idx, 'pred'] = preds_sub['pred'].values

    print("F1 total:", f1_score(df["graph_vertex_id"], final_preds['pred'], average="weighted"))
    print("Accuracy total:", accuracy_score(df["graph_vertex_id"], final_preds['pred']))

    print("finished")
