# Binary Decision Forest for Verilog pmux to mux_tree Conversion

This implementation provides a Binary Decision Forest system to convert Verilog pmux (casez statements) to optimized mux_tree (nested ternary operators) with minimal AIG node count.

## Features

### Core Classes

- **`BinaryDecisionTree`**: Handles individual decision trees for single output bits
- **`BinaryDecisionForest`**: Manages multiple trees and provides optimization capabilities
- **`TreeNode`**: Represents nodes in the binary decision tree with reuse detection

### Key Capabilities

1. **Tree Construction**: Build binary decision trees from sel variables and output truth tables
2. **Simplification**: Automatic tree simplification using rules for don't care values and identical subtrees
3. **AIG Node Counting**: Count total nodes with proper reuse detection across multiple trees
4. **Expression Generation**: Generate mux_tree expressions in Verilog format
5. **Optimization**: Find optimal sel variable ordering to minimize total AIG nodes

### Terminal Value Encoding

- `1`: Output constant 1 (`1'b1`)
- `0`: Output constant 0 (`1'b0`)
- `-1`: Don't care value (`1'bx`)
- `2*i+2`: Output `din[i]`
- `2*i+3`: Output `~din[i]`

### Simplification Rules

From terminal nodes to root:
- Both children are -1 → current node = -1
- Both children same → current node = same value
- Left child = -1 → current node = right child
- Right child = -1 → current node = left child
- Identical subtrees → merge to single subtree

## Usage Example

```python
from mux_forest import BinaryDecisionForest

# Define selection variables (MSB first for tree hierarchy)
sel_vars = ["sel[3]", "sel[2]", "sel[1]", "sel[0]"]

# Define output truth tables for each dout bit
# Length must be 2^len(sel_vars), indexed by binary combination
dout_lists = [
    [-1, -1, -1, -1, -1, 1, -1, -1, 8, -1, -1, 1, -1, 0, -1, -1],  # dout[0]
    [-1, -1, -1, -1, -1, 7, -1, -1, 0, -1, -1, 0, -1, 1, -1, -1],  # dout[1]
    [-1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 0, -1, -1],  # dout[2]
    [-1, -1, -1, -1, -1, 0, -1, -1, 0, -1, -1, 1, -1, 4, -1, -1],  # dout[3]
]

# Create forest
forest = BinaryDecisionForest(sel_vars, dout_lists)

# Generate expressions
expressions = forest.get_expressions()
assigns = forest.generate_verilog_assigns("dout")

# Print results
for assign in assigns:
    print(assign)

# Optimize sel variable ordering
optimal_order, min_nodes = forest.optimize_sel_order()
print(f"Optimal order: {optimal_order}")
print(f"Minimum nodes: {min_nodes}")
```

## Example: Verilog casez Conversion

Input Verilog:
```verilog
always @(*)
    casez (sel)
        4'b1011 : dout = {1'b1, 1'b1, 1'b0, 1'b1};
        4'b0101 : dout = {1'b0, 1'b1, ~din[2], 1'b1};
        4'b1101 : dout = {din[1], 1'b0, 1'b1, 1'b0};
        4'b1000 : dout = {1'b0, 1'b1, 1'b0, din[3]};
        default: dout = {1'bx, 1'bx, 1'bx, 1'bx};
    endcase
```

Output mux_tree (optimized):
```verilog
assign dout[0] = (sel[2] ? (sel[3] ? 1'b0 : 1'b1) : (sel[1] ? 1'b1 : din[3]));
assign dout[1] = (sel[2] ? (sel[3] ? 1'b1 : ~din[2]) : 1'b0);
assign dout[2] = (sel[2] ? (sel[3] ? 1'b0 : 1'b1) : 1'b1);
assign dout[3] = (sel[2] ? (sel[3] ? din[1] : 1'b0) : (sel[1] ? 1'b1 : 1'b0));
```

The optimization reduced AIG node count from 15 to 14 nodes by reordering sel variables.

## Testing

Run the comprehensive test suite:

```bash
python test_comprehensive.py
```

Test specific functionality:

```bash
python test_debug.py        # Basic debugging and verification
python test_optimization.py # Optimization algorithm testing  
python test_optimal.py      # Test optimal ordering results
```

## Implementation Details

### Tree Construction
- Binary decision trees are built recursively with sel variables as internal nodes
- Truth table indices map to binary combinations of sel variables
- Left child represents sel=0, right child represents sel=1

### Node Reuse Detection
- TreeNode implements `__eq__` and `__hash__` for proper equality checking
- Node counting tracks visited nodes to avoid double counting shared subtrees
- Identical subtrees are merged during simplification

### Optimization Algorithm
- Tests all permutations of sel variable orderings (factorial complexity)
- Reorders truth tables according to new variable mappings
- Selects ordering with minimum total AIG node count

### Complexity
- Tree construction: O(n * 2^k) where n = number of outputs, k = number of sel variables
- Optimization: O(k! * n * 2^k) where k! is factorial of sel variable count
- For practical use, optimization is feasible for k ≤ 6 variables