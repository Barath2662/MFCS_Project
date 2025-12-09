import streamlit as st
import numpy as np
from scipy.optimize import linear_sum_assignment # Using scipy's robust implementation
import pandas as pd # Import pandas for DataFrame creation

# --- Configuration ---
st.set_page_config(page_title="Operations Research Solver", layout="wide")
st.title("Operations Research Solver")

# --- Assignment Problem Solver ---
def solve_assignment(cost_matrix, problem_type='min'):
    """
    Solves the Assignment Problem using the Hungarian Algorithm (via scipy).
    Handles both minimization and maximization by transforming the matrix.
    """
    cost_matrix = np.array(cost_matrix, dtype=float)
    original_cost = cost_matrix.copy()

    # 1. Handle Maximization by converting to a minimization problem
    if problem_type == 'max':
        # Find the maximum value and subtract all elements from it.
        max_val = np.max(cost_matrix)
        cost_matrix = max_val - cost_matrix

    # 2. Scipy's linear_sum_assignment handles padding/unbalanced matrix internally
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 3. Calculate the total cost/profit based on the original matrix
    total_original_cost = original_cost[row_ind, col_ind].sum()
    
    # 4. Prepare results for display
    valid_assignments = list(zip(row_ind, col_ind))
    
    return valid_assignments, total_original_cost

# --- Transportation Problem Solvers ---

def balance(cost, supply, demand):
    """Balances the transportation problem by adding a dummy source/destination with zero cost."""
    supply_sum = sum(supply)
    demand_sum = sum(demand)
    
    # Convert to lists of lists for cost and lists for supply/demand
    cost = [row[:] for row in cost]
    supply = supply[:]
    demand = demand[:]

    if supply_sum == demand_sum:
        return cost, supply, demand

    num_sources = len(cost)
    num_destinations = len(cost[0]) if num_sources > 0 else 0

    if supply_sum > demand_sum:  # Add Dummy Destination
        diff = supply_sum - demand_sum
        demand.append(diff)
        # Add a 0 cost to the new dummy destination for all sources
        for i in range(num_sources):
            cost[i].append(0)
    elif demand_sum > supply_sum:  # Add Dummy Source
        diff = demand_sum - supply_sum
        supply.append(diff)
        # Add a new source row with 0 cost to all destinations
        cost.append([0] * num_destinations)

    return cost, supply, demand

def nwc_method(cost, supply, demand):
    """North-West Corner Method for initial basic feasible solution."""
    # Ensure all lists are copies
    cost = [row[:] for row in cost]
    supply = supply[:]
    demand = demand[:]
    m, n = len(cost), len(cost[0])
    alloc = [[0] * n for _ in range(m)]
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
    """Least Cost Method (Matrix Minimum Method) for initial basic feasible solution."""
    # Ensure all lists are copies
    cost_copy = [row[:] for row in cost]
    original_cost = cost # Keep reference to original costs
    supply = supply[:]
    demand = demand[:]
    m, n = len(cost_copy), len(cost_copy[0])
    alloc = [[0] * n for _ in range(m)]

    # Use a large number instead of inf to simplify list-based code
    INF_COST = 1e9  

    while True:
        min_val = INF_COST
        min_cell = None

        # Find the cell with the minimum cost
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

        # Mark exhausted rows/columns with a high cost
        if supply[i] == 0:
            for c in range(n): cost_copy[i][c] = INF_COST
        if demand[j] == 0:
            for r in range(m): cost_copy[r][j] = INF_COST

    total_cost = sum(alloc[i][j] * original_cost[i][j] for i in range(m) for j in range(n))
    return alloc, total_cost

def vam(cost, supply, demand):
    """
    Vogel's Approximation Method (VAM) for initial basic feasible solution.
    """
    import numpy as np
    
    original_cost = np.array(cost, dtype=float)
    current_cost = original_cost.copy()
    current_supply = np.array(supply, dtype=float)
    current_demand = np.array(demand, dtype=float)
    
    m, n = len(current_supply), len(current_demand)
    alloc = np.zeros((m, n), dtype=float)
    
    # Use masks to track active rows and columns
    row_mask = np.ones(m, dtype=bool)
    col_mask = np.ones(n, dtype=bool)
    
    # A large value to represent exhausted/invalid costs
    INF_VAL = np.inf
    
    while np.any(row_mask) and np.any(col_mask):
        
        # --- 1. Calculate Penalties ---
        
        # Function to get the difference between the two smallest costs
        def get_penalty(arr):
            # Only consider active costs
            active_costs = arr[arr != INF_VAL]
            if len(active_costs) >= 2:
                # Use partition to efficiently find the two smallest values
                min_vals = np.partition(active_costs, 1)[:2]
                return min_vals[1] - min_vals[0]
            elif len(active_costs) == 1:
                # If only one cost is left, penalty is 0 (or a very low number)
                return 0.0
            else:
                return -1.0 # Sentinel value for fully allocated

        # Row Penalties
        row_penalties = np.array([get_penalty(current_cost[i]) for i in range(m)])
        row_penalties[~row_mask] = -1.0 # Ignore exhausted rows

        # Column Penalties
        col_penalties = np.array([get_penalty(current_cost[:, j]) for j in range(n)])
        col_penalties[~col_mask] = -1.0 # Ignore exhausted columns
        
        # If all valid penalties are -1, the loop should terminate (e.g., only 1 row/col left)
        if np.max(row_penalties) == -1.0 and np.max(col_penalties) == -1.0:
            break

        # --- 2. Find Max Penalty & Corresponding Cell ---
        
        max_row_penalty = np.max(row_penalties)
        max_col_penalty = np.max(col_penalties)
        
        selected_row = False
        
        if max_row_penalty >= max_col_penalty:
            # Select row with max penalty (or the first if tied)
            i = np.argmax(row_penalties)
            # Find the minimum cost cell in that row (only active columns)
            min_cost_in_row = INF_VAL
            j = -1
            for col_idx in np.where(col_mask)[0]:
                if current_cost[i, col_idx] < min_cost_in_row:
                    min_cost_in_row = current_cost[i, col_idx]
                    j = col_idx
            selected_row = True
        else:
            # Select column with max penalty
            j = np.argmax(col_penalties)
            # Find the minimum cost cell in that column (only active rows)
            min_cost_in_col = INF_VAL
            i = -1
            for row_idx in np.where(row_mask)[0]:
                if current_cost[row_idx, j] < min_cost_in_col:
                    min_cost_in_col = current_cost[row_idx, j]
                    i = row_idx

        # If i or j is still -1, something went wrong, break
        if i == -1 or j == -1:
            break
            
        # --- 3. Allocate ---
        
        allocation = min(current_supply[i], current_demand[j])
        alloc[i, j] = allocation
        
        # Update supply and demand
        current_supply[i] -= allocation
        current_demand[j] -= allocation
        
        # --- 4. Update Masks and Costs ---
        
        # Use a small tolerance for floating point comparisons
        TOL = 1e-9
        
        if current_supply[i] <= TOL:
            row_mask[i] = False
            # Invalidate all costs in the exhausted row for future penalty calculations
            current_cost[i, :] = INF_VAL
            
        if current_demand[j] <= TOL:
            col_mask[j] = False
            # Invalidate all costs in the exhausted column for future penalty calculations
            current_cost[:, j] = INF_VAL
            
    # Calculate total cost using the original cost matrix
    total_cost = np.sum(alloc * original_cost)
    
    return alloc.astype(int).tolist(), float(total_cost) # Return as standard Python types

# --- Main Streamlit Application Logic ---

problem_type = st.sidebar.selectbox("Select Problem Type", ["Assignment Problem", "Transportation Problem"])

if problem_type == "Assignment Problem":
    # ----------------------------------------------------
    # ASSIGNMENT PROBLEM UI
    # ----------------------------------------------------
    st.header("Assignment Problem Solver")
    
    # Problem type and dimensions
    st.subheader("Configuration")
    layout_cols = st.columns(3)
    with layout_cols[0]:
        rows = st.number_input("Number of Rows (e.g., Jobs/Workers)", min_value=1, value=3, step=1, key='assign_rows')
    with layout_cols[1]:
        cols = st.number_input("Number of Columns (e.g., Machines/Tasks)", min_value=1, value=3, step=1, key='assign_cols')
    with layout_cols[2]:
        problem_type_op = st.radio("Objective", ["min", "max"], 
                                   format_func=lambda x: "Minimization (Cost)" if x == "min" else "Maximization (Profit)",
                                   horizontal=True, key='assign_type')
    
    # Create cost matrix input
    st.subheader("Cost/Profit Matrix")
    st.info("Enter the values (Cost for Minimization, Profit for Maximization).")
    cost_matrix = []
    
    # Dynamic inputs for the matrix
    matrix_cols = st.columns(cols)
    for i in range(rows):
        row = []
        for j in range(cols):
            with matrix_cols[j]:
                # Use a specific, consistent key pattern
                row.append(st.number_input(f"Row {i+1} / Col {j+1}", value=int(np.random.randint(1, 20)), key=f"cost_{i}_{j}"))
        cost_matrix.append(row)
    
    st.write("Current Matrix:")
    st.dataframe(np.array(cost_matrix), use_container_width=True)


    st.markdown("---")
    if st.button("Solve Assignment Problem", key="solve_assign_btn"):
        try:
            # Convert cost matrix to numpy array for consistent handling
            cost_array = np.array(cost_matrix)
            assignments, total_cost = solve_assignment(cost_array, problem_type_op)
            
            st.subheader("Optimal Solution")
            objective_word = 'Cost' if problem_type_op == 'min' else 'Profit'

            st.success(f"**Optimal Total {objective_word}: {total_cost:.2f}**")
            
            st.write("**Assignments:**")
            
            # Display results in a table
            data = []
            for i, j in assignments:
                cost_val = cost_array[i, j]
                data.append({
                    "Row (Source)": i + 1,
                    "Column (Destination)": j + 1,
                    f"{objective_word} Value": cost_val
                })
            
            st.dataframe(data, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"An error occurred while solving the assignment problem: {str(e)}")

else:
    # ----------------------------------------------------
    # TRANSPORTATION PROBLEM UI
    # ----------------------------------------------------
    st.header("Transportation Problem Solver")
    
    st.subheader("Configuration")
    cols_config = st.columns(2)
    with cols_config[0]:
        sources = st.number_input("Number of Sources (m)", min_value=1, value=3, step=1, key="sources")
    with cols_config[1]:
        destinations = st.number_input("Number of Destinations (n)", min_value=1, value=3, step=1, key="dests")
    
    # Cost matrix input
    st.subheader("Unit Cost Matrix")
    st.info("Enter the unit transportation costs from each source to each destination.")
    cost_matrix = []
    
    matrix_cols = st.columns(destinations)
    for i in range(sources):
        row = []
        for j in range(destinations):
            with matrix_cols[j]:
                row.append(st.number_input(f"C{i+1}{j+1}", value=int(np.random.randint(1, 10)), key=f"t_cost_{i}_{j}"))
        cost_matrix.append(row)
    
    # Supply and demand inputs
    st.subheader("Supply and Demand")
    
    cols_sd = st.columns(2)
    
    with cols_sd[0]:
        st.write("Supply (from each source):")
        supply = []
        for i in range(sources):
            supply.append(st.number_input(f"Supply S{i+1}", min_value=0, value=20, step=1, key=f"supply_{i}"))
    
    with cols_sd[1]:
        st.write("Demand (at each destination):")
        demand = []
        for j in range(destinations):
            demand.append(st.number_input(f"Demand D{j+1}", min_value=0, value=20, step=1, key=f"demand_{j}"))

    # Check for balance
    st.markdown("---")
    supply_sum = sum(supply)
    demand_sum = sum(demand)
    
    if supply_sum != demand_sum:
        st.warning(f"**Unbalanced Problem!** Total Supply ({supply_sum}) $\\neq$ Total Demand ({demand_sum}). A dummy source/destination will be added automatically to balance it for solving.")
    else:
        st.info(f"**Balanced Problem.** Total Supply = Total Demand = {supply_sum}.")

    # Method selection
    st.markdown("---")
    st.subheader("Initial Basic Feasible Solution Method")
    method = st.radio(
        "",
        ["North-West Corner", "Least Cost Method", "Vogel's Approximation (VAM)"],
        horizontal=True
    )
    
    if st.button(f"Solve using {method}", key="solve_tp_btn"):
        try:
            # 1. Balance the problem
            balanced_cost, balanced_supply, balanced_demand = balance(cost_matrix, supply, demand)
            
            # 2. Solve using selected method
            if method == "North-West Corner":
                alloc, total_cost = nwc_method(balanced_cost, balanced_supply, balanced_demand)
            elif method == "Least Cost Method":
                alloc, total_cost = least_cost(balanced_cost, balanced_supply, balanced_demand)
            else:  # VAM
                # VAM uses the original inputs and handles the balancing logic within its implementation
                alloc, total_cost = vam(cost_matrix, supply, demand) 

            st.subheader(f"Results ({method})")
            
            # Display results
            st.success(f"**Initial Basic Feasible Solution Total Transportation Cost: {total_cost:.2f}**")
            
            st.write("**Allocation Matrix:** (Cells show the quantity allocated)")
            
            # --- FIX: Use Pandas DataFrame for Labeled Output ---
            # Label columns/rows for the balanced matrix
            source_labels = [f"S{i+1}" for i in range(sources)]
            dest_labels = [f"D{j+1}" for j in range(destinations)]
            
            # Determine the labels used for the solved (potentially balanced) matrix
            if len(balanced_supply) > sources:
                source_labels.append("Dummy Source")
            if len(balanced_demand) > destinations:
                dest_labels.append("Dummy Destination")

            # Create the DataFrame with labels
            allocation_df = pd.DataFrame(
                np.array(alloc),
                index=source_labels,
                columns=dest_labels
            )

            st.dataframe(allocation_df, use_container_width=True)

            with st.expander("View Balanced Cost Matrix (with Dummy Source/Dest)"):
                st.write("The matrix used for calculation, showing 0 cost for dummy entries.")
                
                # Get the cost matrix used in the final calculation (either balanced_cost or the one
                # resulting from internal VAM balancing)
                if method == "Vogel's Approximation (VAM)":
                    # Re-balance just for consistent display of the matrix labels
                    display_cost, _, _ = balance(cost_matrix, supply, demand)
                else:
                    display_cost = balanced_cost
                
                cost_df = pd.DataFrame(
                    np.array(display_cost),
                    index=source_labels,
                    columns=dest_labels
                )
                st.dataframe(cost_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"An error occurred while solving the transportation problem: {str(e)}")

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