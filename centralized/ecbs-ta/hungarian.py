import math
from typing import List, Tuple, Set
import numpy as np
from scipy.optimize import linear_sum_assignment


# Hungarian algorithm for global minimal cost
def hungarian(cost_mtx: np.ndarray):
    row_ind, col_ind = linear_sum_assignment(cost_mtx)
    matching = []
    for i, start in enumerate(row_ind):
        matching.append((start, col_ind[i]))
    return matching, cost_mtx[row_ind, col_ind].sum()


def constrained_hungarian(cost_mtx: np.ndarray, I: Set[Tuple], O: Set[Tuple]):
    m_cost_mtx = cost_mtx.copy()
    for o in O:
        m_cost_mtx[o[0], o[1]] = np.inf

    for i in I:
        m_cost_mtx[i[0], i[1]] = -1

    try:
        row_ind, col_ind = linear_sum_assignment(m_cost_mtx)
    except ValueError:
        return [], math.inf
    matching = []
    for i, start in enumerate(row_ind):
        matching.append((start, col_ind[i]))
    return matching, cost_mtx[row_ind, col_ind].sum()


if __name__ == "__main__":
    starts = [(0, 0), (0, 2)]
    goals = [(4, 1), (1, 0), (2, 3)]

    agents = hungarian(starts, goals)
    print(agents)
