import numpy as np
import os.path as osp
import itertools
import collections
from src.causal import _swap_rows_and_cols


def analyze():
    path = '../../../N5_test'
    if not osp.exists(path):
        raise FileNotFoundError("test file not found.")

    data = np.load(path, allow_pickle=True, encoding='bytes')
    dags = data[b'DAGs']

    uniques = set()
    unique_dag_lists = collections.defaultdict(list)
    for i, dag in enumerate(dags):
        for perm in itertools.permutations(range(5), 5):
            permed = _swap_rows_and_cols(dag, perm)
            if tuple(permed.reshape(-1)) in uniques:
                unique_dag_lists[tuple(permed.reshape(-1))].append(dag)
                break
        else:
            uniques.add(tuple(dag.reshape(-1)))

    print(len(uniques))
    print([len(l) for l in unique_dag_lists.values()])


if __name__ == "__main__":
    analyze()
