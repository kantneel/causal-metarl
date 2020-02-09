import numpy as np
import itertools

# paper here: https://arxiv.org/pdf/1901.08162.pdf
# N = 5 nodes, edges in upper triangular matrix from {-1, 0, 1}


class CausalGraph:
    all_adj_lists = list(itertools.product([-1, 0, 1], repeat=10))

    def __init__(self, adj_list):
        """
        Create the causal graph structure.
        :param adj_list: 10 ints in {-1, 0, 1} which form the upper-tri adjacency matrix
        """
        adj_mat = np.zeros((5, 5))
        adj_triu_list = np.triu_indices(5, 1)

        adj_mat[adj_triu_list] = adj_list
        self.adj_mat = adj_mat
        self.nodes = [CausalNode(i, self.adj_mat) for i in range(5)]

        # TODO: get equivalence classes for graphs


class CausalNode:
    def __init__(self, id, adj_mat):
        """
        Create data structure for node which knows its parents
        :param id: index of node in graph
        :param adj_mat: upper triangular matrix for graph
        """
        self.id = id
        parents_row = adj_mat[id]
        self.edges = {i: parents_row[i] for i in range(5) if parents_row[i] != 0}  # parent_id : edge_weight

    def sample(self, cond_data):
        """
        Sample this node given the data in cond_data
        :param cond_data: {node_id : value}
        """
        node_cond_mean = sum(k * v for k, v in cond_data.items() if k in self.edges)
        return np.random.normal(node_cond_mean, 0.1)

