# mux_forest

Binary Decision Forest implementation for optimized Verilog pmux to mux_tree conversion.

## Overview

This project implements a Binary Decision Forest system that converts Verilog pmux (casez statements) to optimized mux_tree (nested ternary operators) with minimal AIG node count. The system supports handling outputs with variables (like `din[i]`) and don't care values.

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
- **Sel Variable Optimization**: Find optimal sel signal ordering to minimize total AIG nodes

## Files

- `mux_forest.py` - Main implementation with `BinaryDecisionTree` and `BinaryDecisionForest` classes
- `test_comprehensive.py` - Complete test suite with unit tests
- `test_debug.py` - Debug utilities and basic verification tests
- `test_optimization.py` - Optimization algorithm testing
- `test_optimal.py` - Test optimal ordering results
- `README_IMPLEMENTATION.md` - Detailed implementation documentation

## Example Results

For the provided Verilog example, the system reduces AIG node count from 15 to 14 nodes through optimal sel variable reordering, generating expressions like:

```verilog
assign dout[0] = (sel[2] ? (sel[3] ? 1'b0 : 1'b1) : (sel[1] ? 1'b1 : din[3]));
assign dout[1] = (sel[2] ? (sel[3] ? 1'b1 : ~din[2]) : 1'b0);
```

See `README_IMPLEMENTATION.md` for complete usage documentation and technical details.
