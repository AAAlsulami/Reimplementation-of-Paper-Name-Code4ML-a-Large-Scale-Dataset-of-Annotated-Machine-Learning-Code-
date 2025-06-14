import pickle
import random
import string

import cloudpickle
import json
import os

import pandas as pd
import numpy as np
import scipy
import ast
import astunparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import f1_score, accuracy_score


GRAPH_PATH = "../data/actual_graph.csv"


def load_data(DATASET_PATH, sep=','):
    df = pd.read_csv(DATASET_PATH, encoding='utf-8', comment='#', sep=sep)
    df.dropna(axis=0, inplace=True)
    return df


def get_graph_vertices(GRAPH_VER):
    GRAPH_DIR = '../graph/graph_v{}.txt'.format(GRAPH_VER)
    with open(GRAPH_DIR, "r") as graph_file:
        graph = json.load(graph_file)
        vertices = list(graph.keys())
    print('vertices parsed: {}'.format(vertices))
    return vertices


def tfidf_transform(corpus, tfidf_params, TFIDF_DIR):
    tfidf = cloudpickle.load(open(TFIDF_DIR, 'rb'))
    features = tfidf.transform(corpus)
    return features


def tfidf_fit_transform(code_blocks: pd.DataFrame, tfidf_params: dict, tfidf_path=None):
    tfidf = TfidfVectorizer(**tfidf_params).fit(code_blocks)
    
    if tfidf_path is not None:
        os.makedirs(os.path.dirname(tfidf_path), exist_ok=True)  # تأكد من وجود المسار
        cloudpickle.dump(tfidf, open(tfidf_path, "wb"))
    
    code_blocks_tfidf = tfidf.transform(code_blocks)
    code_blocks_tfidf.sort_indices()
    return code_blocks_tfidf



def count_transform(code_blocks, countvec_path):
    count_vec = pickle.load(open(countvec_path, "rb"))
    features = count_vec.transform(code_blocks)
    return features


def count_fit_transform(code_blocks, countvec_params, countvec_path=None):
    count_vec = CountVectorizer(**countvec_params).fit(code_blocks)
    if countvec_path is not None:
        pickle.dump(count_vec, open(countvec_path, "wb"))
    counts = count_vec.transform(code_blocks)
    return counts


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def get_metrics(X, y, TAGS_TO_PREDICT, MODEL_DIR):
    clf = pickle.load(open(MODEL_DIR, 'rb'))
    print("the model has been loaded")
    y_pred = clf.predict(X)
    print("predictions were calculated")
    accuracy = clf.score(X, y)
    f1 = f1_score(y_pred, y, average='weighted')
    print(f'Mean Accuracy {round(accuracy*100, 2)}%')
    print(f'F1-score {round(f1*100, 2)}%')
    metrics = {'test_accuracy': accuracy, 'test_f1_score': f1}
    return X, y, y_pred, metrics


def create_path(path):
    path_levels = path.split('/')
    cur_path = ""
    for path_seg in path_levels:
        if len(cur_path):
            cur_path = cur_path + "/" + path_seg
        else:
            cur_path = path_seg
        if not os.path.exists(cur_path):
            os.mkdir(cur_path)


def cross_val_scores(kf, clf, X, y):
    f1s = []
    accuracies = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        f1s.append(f1_score(y_test, y_pred, average="weighted"))
        accuracies.append(accuracy_score(y_test, y_pred))

    f1s = np.array(f1s)
    accuracies = np.array(accuracies)
    return f1s.mean(), f1s.std(), accuracies.mean(), accuracies.std()


def make_tokenizer(model):
    """
    This is supposed to work for tokenizers from huggingface lib
    See: https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#tokenizer
    """
    def tokenizer(s):
        output = model.encode(s)
        return output.tokens

    return tokenizer


class Transformer(ast.NodeTransformer):
    def __init__(self, masking_rate, randomize=False):
        self.vars = dict()
        self.count = 0
        self.masking_rate = masking_rate
        self.generator = np.random.default_rng(42)
        self.randomize = randomize

    def generic_visit(self, node):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            if node.id not in self.vars:
                if self.generator.random() > self.masking_rate:
                    self.vars[node.id] = node.id
                else:
                    if not self.randomize:
                        self.vars[node.id] = "[VAR" + str(self.count) + "]"
                        self.count += 1
                    else:
                        self.vars[node.id] = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(7))
                        self.count += 1
            node.id = self.vars[node.id]
        return node


def mask(row, code_col, masking_rate, randomize=False):
    source = row[code_col]

    try:
        root = ast.parse(source)
        transformer = Transformer(masking_rate, randomize)
        for node in ast.walk(root):
            transformer.visit(node)
        row[code_col] = astunparse.unparse(root)
        return row
    except SyntaxError:
        return row


def augment_mask(dataset, code_col, masking_rate):
    augmented = dataset.copy()
    augmented.apply(lambda row: mask(row, code_col, masking_rate), axis=1)
    return augmented
