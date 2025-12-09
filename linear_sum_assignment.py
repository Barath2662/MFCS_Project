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
covered_rows = [i for i in range(n) if i not in assigned_rows]
covered_cols = assigned_cols.copy()

uncovered = []
for i in range(n):
    for j in range(n):
        if i not in covered_rows and j not in covered_cols:
            uncovered.append(C[i, j])
if uncovered:
    min_uncovered = min(uncovered)
else:
    min_uncovered = 0

for i in range(n):
    for j in range(n):
        if i not in covered_rows and j not in covered_cols:
            C[i, j] -= min_uncovered
        elif i in covered_rows and j in covered_cols:
            C[i, j] += min_uncovered

final_rows = []
final_cols = []
   
for i in range(n):
    for j in range(n):
        if C[i, j] == 0 and i not in final_rows and j not in final_cols:
            final_rows.append(i)
            final_cols.append(j)
        