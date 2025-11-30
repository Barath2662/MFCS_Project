import streamlit as st
import numpy as np
from scipy.optimize import linear_sum_assignment

st.set_page_config(page_title="Operations Research Solver", layout="wide")
st.title("Operations Research Solver")

# Problem type selection
problem_type = st.sidebar.selectbox("Select Problem Type", ["Assignment Problem", "Transportation Problem"])

# Common functions
def solve_assignment(cost_matrix, problem_type='min'):
    cost_matrix = np.array(cost_matrix)
    
    # Convert maximization to minimization problem if needed
    if problem_type == 'max':
        max_val = np.max(cost_matrix)
        cost_matrix = max_val - cost_matrix
    
    r, c = cost_matrix.shape
    size = max(r, c)
    
    # Create a square matrix by adding dummy rows/columns with zero cost
    padded = np.zeros((size, size))
    padded[:r, :c] = cost_matrix
    
    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(padded)
    
    # Filter out dummy assignments (where either row or column is beyond the original matrix dimensions)
    valid_assignments = []
    total_cost = 0
    
    for i, j in zip(row_ind, col_ind):
        if i < r and j < c:  # Only include assignments within the original matrix dimensions
            valid_assignments.append((i, j))
            total_cost += cost_matrix[i, j]
    
    return valid_assignments, total_cost

def balance(cost, supply, demand):
    supply_sum = sum(supply)
    demand_sum = sum(demand)
    cost = [row[:] for row in cost]
    supply = supply[:]
    demand = demand[:]

    if supply_sum == demand_sum:
        return cost, supply, demand

    if supply_sum > demand_sum:  # Add dummy demand
        diff = supply_sum - demand_sum
        demand.append(diff)
        for row in cost:
            row.append(0)
    else:  # Add dummy supply
        diff = demand_sum - supply_sum
        supply.append(diff)
        cost.append([0] * len(demand))

    return cost, supply, demand

def nwc_method(cost, supply, demand):
    cost = [row[:] for row in cost]
    supply = supply[:]
    demand = demand[:]
    m, n = len(cost), len(cost[0])
    alloc = [[0]*n for _ in range(m)]
    i = j = 0

    while i < m and j < n:
        val = min(supply[i], demand[j])
        alloc[i][j] = val
        supply[i] -= val
        demand[j] -= val
        if supply[i] == 0: i += 1
        if demand[j] == 0: j += 1

    total_cost = sum(alloc[i][j] * cost[i][j] for i in range(m) for j in range(n))
    return alloc, total_cost

def least_cost(cost, supply, demand):
    cost_copy = [row[:] for row in cost]
    supply = supply[:]
    demand = demand[:]
    m, n = len(cost_copy), len(cost_copy[0])
    alloc = [[0]*n for _ in range(m)]

    while True:
        min_val = float('inf')
        min_cell = None

        for i in range(m):
            for j in range(n):
                if supply[i] > 0 and demand[j] > 0 and cost_copy[i][j] < min_val:
                    min_val = cost_copy[i][j]
                    min_cell = (i, j)

        if not min_cell:
            break

        i, j = min_cell
        val = min(supply[i], demand[j])
        alloc[i][j] = val
        supply[i] -= val
        demand[j] -= val

        if supply[i] == 0:
            for c in range(n): cost_copy[i][c] = float('inf')
        if demand[j] == 0:
            for r in range(m): cost_copy[r][j] = float('inf')

    total_cost = sum(alloc[i][j] * cost[i][j] for i in range(m) for j in range(n))
    return alloc, total_cost

def vam(cost, supply, demand):
    cost_copy = [row[:] for row in cost]
    supply = supply[:]
    demand = demand[:]
    m, n = len(cost_copy), len(cost_copy[0])
    alloc = [[0]*n for _ in range(m)]

    while sum(supply) > 0:
        penalties = []

        # Row penalties
        for i in range(m):
            if supply[i] > 0:
                row = [cost_copy[i][j] for j in range(n) if demand[j] > 0]
                if len(row) >= 2:
                    s = sorted(row)
                    penalties.append((s[1] - s[0], i, "row"))

        # Column penalties
        for j in range(n):
            if demand[j] > 0:
                col = [cost_copy[i][j] for i in range(m) if supply[i] > 0]
                if len(col) >= 2:
                    s = sorted(col)
                    penalties.append((s[1] - s[0], j, "col"))

        if not penalties:
            break

        penalties.sort(reverse=True)
        _, idx, typ = penalties[0]

        if typ == "row":
            i = idx
            j = min(range(n), key=lambda x: cost_copy[i][x] if demand[x] > 0 else float('inf'))
        else:
            j = idx
            i = min(range(m), key=lambda x: cost_copy[x][j] if supply[x] > 0 else float('inf'))

        val = min(supply[i], demand[j])
        alloc[i][j] = val
        supply[i] -= val
        demand[j] -= val

        if supply[i] == 0:
            for c in range(n): cost_copy[i][c] = float('inf')
        if demand[j] == 0:
            for r in range(m): cost_copy[r][j] = float('inf')

    total_cost = sum(alloc[i][j] * cost[i][j] for i in range(m) for j in range(n))
    return alloc, total_cost

# Main app logic
if problem_type == "Assignment Problem":
    st.header("Assignment Problem")
    
    # Problem type and dimensions
    st.subheader("Problem Type & Dimensions")
    layout_cols = st.columns(3)
    with layout_cols[0]:
        rows = st.number_input("Number of Rows", min_value=1, value=3, step=1, key='assign_rows')
    with layout_cols[1]:
        cols = st.number_input("Number of Columns", min_value=1, value=3, step=1, key='assign_cols')
    with layout_cols[2]:
        problem_type = st.radio("Problem Type", ["min", "max"], 
                              format_func=lambda x: "Minimization" if x == "min" else "Maximization",
                              horizontal=True, key='assign_type')
    
    # Create cost matrix input
    st.subheader("Cost Matrix")
    cost_matrix = []
    for i in range(rows):
        row = []
        input_cols = st.columns(cols)
        for j in range(cols):
            with input_cols[j]:
                row.append(st.number_input(f"C{i+1}{j+1}", value=0, key=f"cost_{i}_{j}"))
        cost_matrix.append(row)
    
    if st.button("Solve Assignment Problem"):
        try:
            assignments, total_cost = solve_assignment(cost_matrix, problem_type)
            st.subheader("Results")
            st.write("Assignments (row → col):")
            for assign in assignments:
                st.write(f"Row {assign[0]+1} → Column {assign[1]+1} (Cost: {cost_matrix[assign[0], assign[1]]})")
            st.success(f"Total {'Cost' if problem_type == 'min' else 'Profit'}: {total_cost}")
            st.info(f"Problem Type: {'Minimization' if problem_type == 'min' else 'Maximization'}")
        except Exception as e:
            st.error(f"Error solving assignment problem: {str(e)}")

else:  # Transportation Problem
    st.header("Transportation Problem")
    
    # Get problem dimensions
    cols = st.columns(2)
    with cols[0]:
        sources = st.number_input("Number of Sources", min_value=1, value=3, step=1, key="sources")
    with cols[1]:
        destinations = st.number_input("Number of Destinations", min_value=1, value=3, step=1, key="dests")
    
    # Cost matrix input
    st.subheader("Cost Matrix")
    cost_matrix = []
    for i in range(sources):
        row = []
        cols = st.columns(destinations)
        for j in range(destinations):
            with cols[j]:
                row.append(st.number_input(f"C{i+1}{j+1}", value=0, key=f"cost_{i}_{j}"))
        cost_matrix.append(row)
    
    # Supply and demand inputs
    st.subheader("Supply and Demand")
    cols = st.columns(2)
    
    with cols[0]:
        st.write("Supply (for each source):")
        supply = []
        for i in range(sources):
            supply.append(st.number_input(f"Supply {i+1}", min_value=0, value=10, key=f"supply_{i}"))
    
    with cols[1]:
        st.write("Demand (for each destination):")
        demand = []
        for j in range(destinations):
            demand.append(st.number_input(f"Demand {j+1}", min_value=0, value=10, key=f"demand_{j}"))
    
    # Method selection
    method = st.radio(
        "Select Method",
        ["North-West Corner", "Least Cost Method", "Vogel's Approximation (VAM)"]
    )
    
    if st.button(f"Solve using {method}"):
        try:
            # Balance the problem if needed
            balanced_cost, balanced_supply, balanced_demand = balance(cost_matrix, supply, demand)
            
            # Solve using selected method
            if method == "North-West Corner":
                alloc, total_cost = nwc_method(balanced_cost, balanced_supply, balanced_demand)
            elif method == "Least Cost Method":
                alloc, total_cost = least_cost(balanced_cost, balanced_supply, balanced_demand)
            else:  # VAM
                alloc, total_cost = vam(balanced_cost, balanced_supply, balanced_demand)
            
            # Display results
            st.subheader("Results")
            st.write("Allocation Matrix:")
            st.write(alloc)
            st.success(f"Total Transportation Cost: {total_cost}")
            
        except Exception as e:
            st.error(f"Error solving transportation problem: {str(e)}")

# Add some styling
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        margin: 10px 0;
    }
    .stRadio>div {
        flex-direction: row;
        gap: 20px;
    }
</style>
""", unsafe_allow_html=True)
