# Operations Research Solver

A Streamlit-based web application for solving Assignment and Transportation problems in Operations Research.

## Features

### 1. Assignment Problem Solver
- Supports both **Minimization** and **Maximization** problems
- Solves balanced and unbalanced assignment problems
- Uses the Hungarian Algorithm for optimal assignments
- Displays individual assignment costs and total value

### 2. Transportation Problem Solver
- Solves balanced and unbalanced transportation problems
- Multiple solution methods:
  - North-West Corner Method (NWC)
  - Least Cost Method (LCM)
  - Vogel's Approximation Method (VAM)
- Automatically handles unbalanced problems by adding dummy sources/destinations

## Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Barath2662/MFCS_Project.git
   cd MFCS_Project
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
streamlit run or_solver.py
```

The application will open in your default web browser at `http://localhost:8501`.

## How to Use

### Assignment Problem
1. Select "Assignment Problem" from the sidebar
2. Enter the number of rows and columns
3. Choose between Minimization or Maximization
4. Fill in the cost/profit matrix
5. Click "Solve Assignment Problem" to see the results

### Transportation Problem
1. Select "Transportation Problem" from the sidebar
2. Enter the number of sources and destinations
3. Fill in the cost matrix, supply, and demand values
4. Choose a solution method (NWC, LCM, or VAM)
5. Click the corresponding button to see the solution

## Example Problems

### Assignment Problem Example (Minimization)
```
Cost Matrix:
[10, 20, 30]
[15, 25, 35]
[20, 30, 40]
```

### Transportation Problem Example
```
Cost Matrix:
[3, 1, 7, 4]
[2, 6, 5, 9]
[8, 3, 3, 2]

Supply: [300, 400, 500]
Demand: [250, 350, 400, 200]
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Built with [Streamlit](https://streamlit.io/)
- Uses [SciPy](https://www.scipy.org/) for optimization algorithms
- [NumPy](https://numpy.org/) for numerical operations
