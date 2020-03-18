import numpy as np
import itertools

# paper here: https://arxiv.org/pdf/1901.08162.pdf
# N = 5 nodes, edges in upper triangular matrix from {-1, 0, 1}


ALL_ADJ_LISTS = [tuple(l) for l in itertools.product([-1, 0, 1], repeat=10)]


def _get_random_adj_list(train):
    idx = np.random.randint(0, len(ALL_ADJ_LISTS))
    return ALL_ADJ_LISTS[idx]


def _swap_rows_and_cols(arr_original, permutation):
    if not isinstance(permutation, list):
        permutation = list(permutation)
    arr = arr_original.copy()
    arr[:] = arr[permutation]
    arr[:, :] = arr[:, permutation]
    return arr


def get_permuted_adj_mats(adj_list):
    adj_mat = np.zeros((5, 5))
    adj_triu_list = np.triu_indices(5, 1)
    adj_mat[adj_triu_list] = adj_list
    perms = set()

    for perm in itertools.permutations(np.arange(5), 5):
        permed = _swap_rows_and_cols(adj_mat, perm)
        if not any(permed[4]):
            perms.add(tuple(permed.reshape(-1)))

    return perms


def true_separate_train_and_test():
    test = set()
    adj_lists_copy = list(ALL_ADJ_LISTS)
    while True:
        idx = np.random.randint(0, len(adj_lists_copy))
        perms = get_permuted_adj_mats(adj_lists_copy[idx])
        test.update(perms)
        print(len(test))
        if len(test) > 408:
            break


class CausalGraph:
    def __init__(self, adj_list=None, train=True, permute=False):
        """
        Create the causal graph structure.
        :param adj_list: 10 ints in {-1, 0, 1} which form the upper-tri adjacency matrix
        """
        if adj_list is None:
            adj_list = _get_random_adj_list(train)

        adj_mat = np.zeros((5, 5))
        adj_triu_list = np.triu_indices(5, 1)

        adj_mat[adj_triu_list] = adj_list
        permutation = np.arange(5)
        if permute:
            permutation = np.random.permutation(permutation)
            adj_mat = _swap_rows_and_cols(adj_mat, permutation)

        self.adj_mat = adj_mat
        self.permutation = permutation
        self.nodes = [CausalNode(i, self.adj_mat) for i in range(5)]

        # TODO: get equivalence classes for graphs

    def intervene(self, node_idx, val):
        """
        Intervene on the node at node_idx by setting its value to val.
        :param node_idx: (int) node to intervene on
        :param val: (float) value to set
        """
        self.nodes[node_idx].intervene(val)

    def sample_all(self):
        """
        Sample all nodes according to their causal relations
        :return: sampled_vals (np.ndarray) array of sampled values
        """
        sampled_vals = np.zeros(5)
        cond_data = dict()
        for node_idx in self.permutation[::-1]:
            # traverse the nodes in topologically sorted order
            node_sample = self.nodes[node_idx].sample(cond_data)
            sampled_vals[node_idx] = node_sample
            cond_data[node_idx] = node_sample

        return sampled_vals

    def get_value(self, node_idx):
        """
        Get the value at index node_idx
        :param node_idx: (int)
        :return: val (float)
        """
        return self.nodes[node_idx].val


class CausalNode:
    def __init__(self, idx, adj_mat):
        """
        Create data structure for node which knows its parents
        :param idx: index of node in graph
        :param adj_mat: upper triangular matrix for graph
        """
        self.id = idx
        self.val = None
        self.intervened = False

        parents_row = adj_mat[idx]
        self.edges = {i: parents_row[i] for i in range(5) if parents_row[i] != 0}  # parent_id : edge_weight

    def sample(self, cond_data):
        """
        Sample this node given the data in cond_data. Resets intervened value afterward
        :param cond_data: {node_id : value}
        """
        if not self.intervened:
            node_cond_mean = sum(self.edges[k] * v for k, v in cond_data.items() if k in self.edges)
            self.val = np.random.normal(node_cond_mean, 0.1)

        self.intervened = False
        return self.val

    def intervene(self, val):
        """
        Intervene on this node. Valid for one call to sample()
        :param val: (float) value to set this node to
        """
        self.val = val
        self.intervened = True
