import numpy as np

def hungarian_solution(cost_matrix):
    C = np.array(cost_matrix, dtype=float)
    n = C.shape[0]

    for i in range(n):
        C[i, :] -= np.min(C[i, :])

    for j in range(n):
        C[:, j] -= np.min(C[:, j])

    while True:
        assigned_rows = []
        assigned_cols = []
       
        for i in range(n):
            for j in range(n):
                if C[i, j] == 0 and i not in assigned_rows and j not in assigned_cols:
                    assigned_rows.append(i)
                    assigned_cols.append(j)

        if len(assigned_rows) == n:
            break
       
        s_min = np.min(C[C > 0])
       
        if s_min is not np.ma.masked and s_min > 0:
            C -= s_min

    final_rows = []
    final_cols = []
   
    for i in range(n):
        for j in range(n):
            if C[i, j] == 0 and i not in final_rows and j not in final_cols:
                final_rows.append(i)
                final_cols.append(j)
   
    return np.array(final_rows), np.array(final_cols)

