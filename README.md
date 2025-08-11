# mux_forest

Binary Decision Forest implementation for optimized Verilog pmux to mux_tree conversion with **independent per-tree optimization**.

## Overview

This project implements a Binary Decision Forest system that converts Verilog pmux (casez statements) to optimized mux_tree (nested ternary operators) with minimal AIG node count. The system supports handling outputs with variables (like `din[i]`) and don't care values, and includes advanced optimization algorithms that **optimize each tree independently** for maximum simplification.

## Key Innovation: Independent Per-Tree Optimization

Unlike traditional approaches that force all trees to use the same sel variable order, this implementation allows **each output bit to find its own optimal sel order**. This leads to:

- **Better individual tree simplification** based on terminal node patterns
- **Natural sharing** when trees happen to have similar optimal structures  
- **Superior overall AIG node reduction** compared to forced common ordering

## Quick Start

```python
from mux_forest import BinaryDecisionForest, parse_verilog_example

# Parse the example from problem statement
sel_vars, dout_lists = parse_verilog_example()

# Create and optimize forest
forest = BinaryDecisionForest(sel_vars, dout_lists)
optimal_orders, min_nodes = forest.optimize_sel_order()  # Returns orders per tree

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
- **Independent Per-Tree Optimization**: Each tree finds its own optimal sel variable ordering
  - Exhaustive search for small problems (≤6 variables)
  - Heuristic optimization for large problems (>6 variables)
  - Variable importance analysis per tree
  - Local search with adjacent swaps and position moves
  - Random restart search with importance-biased sampling

## Optimization Performance

The optimization algorithms automatically adapt to problem size and optimize each tree independently:

| Problem Size | Method | Time | Quality |
|-------------|--------|------|---------|
| ≤6 variables | Exhaustive search per tree | Seconds | Optimal per tree |
| 7-10 variables | Heuristic optimization per tree | Minutes | Near-optimal per tree |
| ≥11 variables | Scalable heuristics per tree | Configurable | Good solutions per tree |

**Benefits of Independent Optimization:**
- Each tree achieves its own optimal structure
- No forced common ordering constraints
- Better overall AIG node reduction

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

## API Changes: Independent Optimization

**Important**: The `optimize_sel_order()` method now returns per-tree orders instead of a single global order:

```python
# OLD API (forced same order for all trees)
optimal_order, min_nodes = forest.optimize_sel_order()  # Returns List[str]

# NEW API (independent order per tree) 
optimal_orders, min_nodes = forest.optimize_sel_order()  # Returns List[List[str]]

# Each tree can now have its own optimal order
for i, order in enumerate(optimal_orders):
    print(f"Tree {i} optimal order: {order}")
```

This change enables:
- **Better individual tree simplification** - each tree finds its optimal structure
- **Natural sharing** - trees with similar patterns naturally converge to similar orders
- **Superior total optimization** - no artificial constraints force suboptimal compromises

## Example Results

For the provided Verilog example, each tree can now find its own optimal sel variable ordering. This typically maintains or improves upon the previous optimization results while allowing much better optimization for cases where trees have different optimal structures.

For large-scale problems (12-bit selection), the independent optimization can achieve significant per-tree improvements, with the system automatically finding the best order for each output bit.

## Documentation

- See `README_IMPLEMENTATION.md` for complete usage documentation and technical details
- See `README_OPTIMIZATION.md` for comprehensive optimization algorithm guide
- Run `python test_large_case.py` to test optimization on large-scale examples
- Run `python test_independent_optimization.py` to see independent optimization in action
