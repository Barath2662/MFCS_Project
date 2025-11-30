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
    # Make copies to avoid modifying the original data
    cost = [row[:] for row in cost]
    supply = supply[:]
    demand = demand[:]
    
    m, n = len(supply), len(demand)
    alloc = [[0] * n for _ in range(m)]
    total_cost = 0
    
    # Balance the problem if needed
    total_supply = sum(supply)
    total_demand = sum(demand)
    
    if total_supply > total_demand:
        demand.append(total_supply - total_demand)
        for row in cost:
            row.append(0)
    elif total_demand > total_supply:
        supply.append(total_demand - total_supply)
        cost.append([0] * len(demand))
    
    m, n = len(supply), len(demand)
    
    while sum(supply) > 0 and sum(demand) > 0:
        # Calculate penalties
        row_penalties = []
        for i in range(m):
            if supply[i] == 0:
                row_penalties.append(-1)
                continue
            row = [cost[i][j] for j in range(n) if demand[j] > 0]
            if len(row) >= 2:
                row_sorted = sorted(row)
                row_penalties.append(row_sorted[1] - row_sorted[0])
            else:
                row_penalties.append(-1)
        
        col_penalties = []
        for j in range(n):
            if demand[j] == 0:
                col_penalties.append(-1)
                continue
            col = [cost[i][j] for i in range(m) if supply[i] > 0]
            if len(col) >= 2:
                col_sorted = sorted(col)
                col_penalties.append(col_sorted[1] - col_sorted[0])
            else:
                col_penalties.append(-1)
        
        # Find maximum penalty
        max_row_penalty = max(row_penalties)
        max_col_penalty = max(col_penalties)
        
        if max_row_penalty >= max_col_penalty:
            # Select row with maximum penalty
            i = row_penalties.index(max_row_penalty)
            # Find minimum cost in this row
            min_cost = float('inf')
            j_min = -1
            for j in range(n):
                if demand[j] > 0 and cost[i][j] < min_cost:
                    min_cost = cost[i][j]
                    j_min = j
            j = j_min
        else:
            # Select column with maximum penalty
            j = col_penalties.index(max_col_penalty)
            # Find minimum cost in this column
            min_cost = float('inf')
            i_min = -1
            for i in range(m):
                if supply[i] > 0 and cost[i][j] < min_cost:
                    min_cost = cost[i][j]
                    i_min = i
            i = i_min
        
        # Allocate as much as possible
        allocation = min(supply[i], demand[j])
        alloc[i][j] = allocation
        total_cost += allocation * cost[i][j]
        
        # Update supply and demand
        supply[i] -= allocation
        demand[j] -= allocation
        
        # If supply is exhausted, set costs to infinity
        if supply[i] == 0:
            for j in range(n):
                cost[i][j] = float('inf')
        
        # If demand is satisfied, set costs to infinity
        if demand[j] == 0:
            for i in range(m):
                cost[i][j] = float('inf')
    
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
