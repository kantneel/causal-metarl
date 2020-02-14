import numpy as np
import pytest

from src.causal import CausalGraph

# forgive the spacing, it's nice for upper triangular settings.
adj_list1 = [0, -1, 0, 0,
                0, -1, 0,
                   0, -1,
                      -1]


def test_1():
    """Create a graph:
        4 -> 3 -> 1
          -> 2 -> 0
    - Test to see that mean observed values are correct
    """
    graph = CausalGraph(adj_list1)

    results = np.zeros(5)
    for i in range(10000):
        graph.intervene(4, 2)
        graph.intervene(2, 5)
        results += graph.sample_all()

    mean = results / 10000
    print(mean)


if __name__ == "__main__":
    test_1()