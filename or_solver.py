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
    original_cost = cost_matrix.copy()
    
    # Convert maximization to minimization problem if needed
    if problem_type == 'max':
        max_val = np.max(cost_matrix) + 1  # Add 1 to ensure all values remain positive
        cost_matrix = max_val - cost_matrix
    
    r, c = cost_matrix.shape
    size = max(r, c)
    
    # Create a square matrix by adding dummy rows/columns with zero cost
    padded = np.zeros((size, size))
    padded[:r, :c] = cost_matrix
    
    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(padded)
    
    # Filter out dummy assignments and calculate total from original costs
    valid_assignments = []
    total_original_cost = 0
    
    for i, j in zip(row_ind, col_ind):
        if i < r and j < c:  # Only include assignments within the original matrix dimensions
            valid_assignments.append((i, j))
            total_original_cost += original_cost[i, j]
    
    return valid_assignments, total_original_cost

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
    import numpy as np
    
    # Convert to numpy arrays for better performance
    cost = np.array(cost, dtype=float)
    supply = np.array(supply, dtype=float)
    demand = np.array(demand, dtype=float)
    
    # Balance the problem if needed
    total_supply = np.sum(supply)
    total_demand = np.sum(demand)
    
    if total_supply > total_demand:
        demand = np.append(demand, total_supply - total_demand)
        cost = np.column_stack((cost, np.zeros(cost.shape[0])))
    elif total_demand > total_supply:
        supply = np.append(supply, total_demand - total_supply)
        cost = np.vstack((cost, np.zeros(cost.shape[1])))
    
    m, n = len(supply), len(demand)
    alloc = np.zeros((m, n))
    total_cost = 0
    
    # Create masks for valid rows and columns
    row_mask = np.ones(m, dtype=bool)
    col_mask = np.ones(n, dtype=bool)
    
    while np.any(row_mask) and np.any(col_mask):
        # Calculate row penalties
        row_penalties = np.full(m, -1.0)
        for i in np.where(row_mask)[0]:
            valid_cols = np.where(col_mask)[0]
            if len(valid_cols) < 2:
                continue
            row_vals = cost[i, valid_cols]
            min_vals = np.partition(row_vals, 1)[:2]  # Get two smallest values
            row_penalties[i] = min_vals[1] - min_vals[0]
        
        # Calculate column penalties
        col_penalties = np.full(n, -1.0)
        for j in np.where(col_mask)[0]:
            valid_rows = np.where(row_mask)[0]
            if len(valid_rows) < 2:
                continue
            col_vals = cost[valid_rows, j]
            min_vals = np.partition(col_vals, 1)[:2]  # Get two smallest values
            col_penalties[j] = min_vals[1] - min_vals[0]
        
        # Find maximum penalty
        max_row_penalty = np.max(row_penalties)
        max_col_penalty = np.max(col_penalties)
        
        if max_row_penalty >= max_col_penalty and max_row_penalty > 0:
            # Select row with maximum penalty
            i = np.argmax(row_penalties)
            # Find minimum cost in this row
            j = np.argmin(np.where(col_mask, cost[i], np.inf))
        else:
            # Select column with maximum penalty
            j = np.argmax(col_penalties)
            # Find minimum cost in this column
            i = np.argmin(np.where(row_mask, cost[:, j], np.inf))
        
        # Allocate as much as possible
        allocation = min(supply[i], demand[j])
        alloc[i, j] = allocation
        total_cost += allocation * cost[i, j]
        
        # Update supply and demand
        supply[i] -= allocation
        demand[j] -= allocation
        
        # Update masks
        if supply[i] <= 1e-10:  # Account for floating point precision
            row_mask[i] = False
            cost[i] = np.inf
        if demand[j] <= 1e-10:  # Account for floating point precision
            col_mask[j] = False
            cost[:, j] = np.inf
    
    # Convert back to Python lists for compatibility
    return alloc.tolist(), float(total_cost)

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
            # Convert cost matrix to numpy array for consistent handling
            cost_array = np.array(cost_matrix)
            assignments, total_cost = solve_assignment(cost_array, problem_type)
            
            st.subheader("Results")
            st.write("Assignments (row → col):")
            
            # Display each assignment with its original cost
            for i, j in assignments:
                cost = cost_array[i, j]
                st.write(f"Row {i+1} → Column {j+1} ({'Cost' if problem_type == 'min' else 'Profit'}: {cost})")
            
            st.success(f"Total {'Cost' if problem_type == 'min' else 'Profit'}: {total_cost}")
            st.info(f"Problem Type: {'Minimization' if problem_type == 'min' else 'Maximization'}")
            
            # Show the original cost matrix for reference
            with st.expander("View Cost/Profit Matrix"):
                st.write("Original Matrix:")
                st.dataframe(cost_array)
                st.write("Note: Values shown are the original input values.")
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
