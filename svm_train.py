import argparse
import logging
import sys
import os
import optuna
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from common.tools import *  # Make sure to implement or adjust this as necessary
import pickle

# Initialize Optuna logging
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("GRAPH_VER", help="Version of the graph you want regex to label your CSV with", type=str)
parser.add_argument("DATASET_PATH", help="Path to your input CSV", type=str)
parser.add_argument("N_TRIALS", help="Optuna n trials, if 0 use default hyperparams", type=int)
args = parser.parse_args()

GRAPH_VER = args.GRAPH_VER
DATASET_PATH = args.DATASET_PATH
N_TRIALS = args.N_TRIALS

MODEL_DIR = f"models/svm_linear_search_graph_v{GRAPH_VER}.sav"
TFIDF_DIR = f"models/tfidf_hyper_svm_graph_v{GRAPH_VER}.pickle"

TAGS_TO_PREDICT = ["dummy_tag"]  # Placeholder for labels to predict

EXPERIMENT_DATA_PATH = ".."
CODE_COLUMN = "code_block"
TARGET_COLUMN = "graph_vertex_id"

RANDOM_STATE = 42
MAX_ITER = 10000

# Hyperparameter search space for Optuna
HYPERPARAM_SPACE = {
    "svm_c": (1e-1, 1e3),
    "tfidf_min_df": (1, 10),
    "tfidf_max_df": (0.2, 0.7),
    "svm_kernel": ["linear", "poly"],
    "svm_degree": (2, 6),
}

DEFAULT_HYPERPARAMS = {
    "svm__C": 37.17,
    "tfidf__min_df": 2,
    "tfidf__max_df": 0.31,
    "svm__kernel": "linear",
    "tfidf__smooth_idf": True,
    "svm__random_state": RANDOM_STATE,
    "svm__max_iter": MAX_ITER,
}


# DEFAULT_HYPERPARAMS = {
#     "svm__C": 1.43,
#     "tfidf__min_df": 6,
#     "tfidf__max_df": 0.30,
#     "svm__kernel": "poly",
#     "svm__degree": 3,
#     "tfidf__smooth_idf": True,
#     "svm__random_state": RANDOM_STATE,
#     "svm__max_iter": MAX_ITER,
# }
#
# DEFAULT_HYPERPARAMS = {
#     "svm__C": 8.71,
#     "tfidf__min_df": 7,
#     "tfidf__max_df": 0.39,
#     "svm__kernel": "rbf",
#     "tfidf__smooth_idf": True,
#     "svm__random_state": RANDOM_STATE,
#     "svm__max_iter": MAX_ITER,
# }


class Objective:
    def __init__(self, df, kfold_params, svm_c, tfidf_min_df, tfidf_max_df, svm_kernel, svm_degree):
        self.kf = KFold(**kfold_params)
        self.c_range = svm_c
        self.min_df_range = tfidf_min_df
        self.max_df_range = tfidf_max_df
        self.kernels = svm_kernel
        self.poly_degrees = svm_degree
        self.df = df

    def __call__(self, trial):
        # TF-IDF parameters
        tfidf_params = {
            "min_df": trial.suggest_int("tfidf__min_df", *self.min_df_range),
            "max_df": trial.suggest_float("tfidf__max_df", *self.max_df_range),
            "smooth_idf": True,
        }
        # Apply TF-IDF transformation
        code_blocks_tfidf = tfidf_fit_transform(self.df[CODE_COLUMN], tfidf_params)
        X, y = code_blocks_tfidf, self.df[TARGET_COLUMN].values

        # SVM parameters
        svm_params = {
            "C": trial.suggest_float("svm__C", *self.c_range, log=True),
            "kernel": trial.suggest_categorical("svm__kernel", self.kernels),
            "random_state": RANDOM_STATE,
            "max_iter": MAX_ITER,
        }
        if svm_params["kernel"] == "poly":
            svm_params["degree"] = trial.suggest_int("svm__degree", *self.poly_degrees)

        clf = SVC(**svm_params)

        # Cross-validation with f1-score
        f1_mean, _, _, _ = cross_val_scores(self.kf, clf, X, y)
        return f1_mean


def select_hyperparams(df, kfold_params, tfidf_path, model_path):
    study = optuna.create_study(direction="maximize", study_name="svm with kernels")
    objective = Objective(df, kfold_params, **HYPERPARAM_SPACE)

    if N_TRIALS > 0:
        study.optimize(objective, n_trials=N_TRIALS)
        params = study.best_params
    else:
        params = DEFAULT_HYPERPARAMS

    # Final SVM and TF-IDF parameters
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

    # Apply the best TF-IDF parameters
    code_blocks_tfidf = tfidf_fit_transform(df[CODE_COLUMN], best_tfidf_params, tfidf_path)
    X, y = code_blocks_tfidf, df[TARGET_COLUMN].values
    clf = SVC(**best_svm_params)

    # Cross-validation metrics
    f1_mean, f1_std, accuracy_mean, accuracy_std = cross_val_scores(objective.kf, clf, X, y)

    # Train on the full dataset and save the model
    clf.fit(X, y)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    pickle.dump(clf, open(model_path, "wb"))

    metrics = dict(
        test_f1_score=f1_mean,
        test_accuracy=accuracy_mean,
        test_f1_std=f1_std,
        test_accuracy_std=accuracy_std,
    )

    return best_tfidf_params, best_svm_params, metrics


if __name__ == "__main__":
    # Load data
    df = load_data(DATASET_PATH)
    print(df.columns)
    nrows = df.shape[0]
    print("loaded")

    kfold_params = {
        "n_splits": 10,
        "random_state": RANDOM_STATE,
        "shuffle": True,
    }

    # Hyperparameter selection and model training
    print("Selecting hyperparameters")
    tfidf_params, svm_params, metrics = select_hyperparams(df, kfold_params, TFIDF_DIR, MODEL_DIR)

    # Print out results
    print("Hyperparameters:", "\ntfidf", tfidf_params, "\nmodel", svm_params)
    print("Metrics:", metrics)
    print("Finished")
