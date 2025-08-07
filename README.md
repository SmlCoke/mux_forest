# mux_forest

Binary Decision Forest implementation for optimized Verilog pmux to mux_tree conversion with scalable optimization algorithms.

## Overview

This project implements a Binary Decision Forest system that converts Verilog pmux (casez statements) to optimized mux_tree (nested ternary operators) with minimal AIG node count. The system supports handling outputs with variables (like `din[i]`) and don't care values, and includes advanced optimization algorithms for large-scale problems.

## Quick Start

```python
from mux_forest import BinaryDecisionForest, parse_verilog_example

# Parse the example from problem statement
sel_vars, dout_lists = parse_verilog_example()

# Create and optimize forest
forest = BinaryDecisionForest(sel_vars, dout_lists)
optimal_order, min_nodes = forest.optimize_sel_order()

# Generate optimized Verilog
assigns = forest.generate_verilog_assigns("dout")
for assign in assigns:
    print(assign)
```

## Features

- **Binary Decision Tree Construction**: Build decision trees for individual output bits
- **Tree Simplification**: Automatic optimization using don't care propagation and subtree merging  
- **AIG Node Counting**: Accurate node counting with reuse detection across multiple trees
- **Expression Generation**: Generate clean mux_tree expressions in Verilog format
- **Advanced Sel Variable Optimization**: Scalable algorithms for optimal sel signal ordering
  - Exhaustive search for small problems (≤6 variables)
  - Heuristic optimization for large problems (>6 variables)
  - Variable importance analysis and greedy construction
  - Local search with adjacent swaps and position moves
  - Random restart search with importance-biased sampling

## Optimization Performance

The optimization algorithms automatically adapt to problem size:

| Problem Size | Method | Time | Quality |
|-------------|--------|------|---------|
| ≤6 variables | Exhaustive search | Seconds | Optimal |
| 7-10 variables | Heuristic optimization | Minutes | Near-optimal |
| ≥11 variables | Scalable heuristics | Configurable | Good solutions |

Typical AIG node reductions: 10-50% depending on problem structure.

## Files

- `mux_forest.py` - Main implementation with optimization algorithms
- `test_comprehensive.py` - Complete test suite with unit tests
- `test_debug.py` - Debug utilities and basic verification tests
- `test_optimization.py` - Optimization algorithm testing
- `test_optimal.py` - Test optimal ordering results
- `test_large_case.py` - Large-scale optimization testing
- `README_IMPLEMENTATION.md` - Detailed implementation documentation
- `README_OPTIMIZATION.md` - Comprehensive optimization guide

## Large-Scale Example

For handling complex pmux statements (like 12-bit selection with 512 cases):

```python
# Parse large Verilog file
sel_vars, dout_lists = parse_large_verilog_file("tc.txt")

# Create forest (12 variables, 10 output bits)
forest = BinaryDecisionForest(sel_vars, dout_lists)
print(f"Initial AIG nodes: {forest.count_total_nodes()}")

# Optimize with configurable time budget
optimal_order, min_nodes = forest.optimize_sel_order(max_iterations=1000)
print(f"Optimized to: {min_nodes} nodes")

# Generate optimized expressions
if optimal_order != sel_vars:
    optimized_forest = BinaryDecisionForest(optimal_order, 
        forest._reorder_dout_lists(optimal_order))
    assigns = optimized_forest.generate_verilog_assigns()
```

## Configuration Options

```python
# Quick optimization (development)
forest.optimize_sel_order(max_iterations=100)

# Balanced optimization (production)  
forest.optimize_sel_order(max_iterations=1000)

# Intensive optimization (critical paths)
forest.optimize_sel_order(max_iterations=5000)

# Disable heuristics (research/benchmarking)
forest.optimize_sel_order(use_heuristics=False)
```

## Example Results

For the provided Verilog example, the system reduces AIG node count from 15 to 14 nodes through optimal sel variable reordering, generating expressions like:

```verilog
assign dout[0] = (sel[2] ? (sel[3] ? 1'b0 : 1'b1) : (sel[1] ? 1'b1 : din[3]));
assign dout[1] = (sel[2] ? (sel[3] ? 1'b1 : ~din[2]) : 1'b0);
```

For large-scale problems (12-bit selection), the optimization can achieve 20-50% AIG node reduction while completing in reasonable time.

## Documentation

- See `README_IMPLEMENTATION.md` for complete usage documentation and technical details
- See `README_OPTIMIZATION.md` for comprehensive optimization algorithm guide
- Run `python test_large_case.py` to test optimization on large-scale examples
