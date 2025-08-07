"""
Binary Decision Forest Implementation for Verilog pmux to mux_tree conversion.

This module implements a system to convert Verilog pmux (casez statements) to 
mux_tree (nested ternary operators) with optimized AIG node count.
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from dataclasses import dataclass
import itertools


@dataclass
class TreeNode:
    """Represents a node in the binary decision tree."""
    value: Optional[int] = None  # Terminal value or None for internal nodes
    left: Optional['TreeNode'] = None  # Left child (sel=0)
    right: Optional['TreeNode'] = None  # Right child (sel=1)
    sel_var: Optional[str] = None  # Selection variable for this node
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node."""
        return self.left is None and self.right is None
    
    def __eq__(self, other) -> bool:
        """Check equality for node reuse detection."""
        if not isinstance(other, TreeNode):
            return False
        if self.is_terminal() and other.is_terminal():
            return self.value == other.value
        elif not self.is_terminal() and not other.is_terminal():
            return (self.sel_var == other.sel_var and 
                   self.left == other.left and 
                   self.right == other.right)
        return False
    
    def __hash__(self) -> int:
        """Hash for node reuse detection."""
        if self.is_terminal():
            return hash(('terminal', self.value))
        else:
            return hash(('internal', self.sel_var, self.left, self.right))


class BinaryDecisionTree:
    """
    Binary Decision Tree for a single output bit.
    
    Handles construction, simplification, and expression generation for one dout bit.
    """
    
    def __init__(self, sel_vars: List[str], dout_values: List[int]):
        """
        Initialize binary decision tree.
        
        Args:
            sel_vars: Selection variables in order (index 0 = highest level/root)
            dout_values: Output values for each combination, length must be 2^len(sel_vars)
                        Encoding: 1=output1, 0=output0, -1=don't care, 
                                 2*i+2=din[i], 2*i+3=~din[i]
        """
        if len(dout_values) != 2 ** len(sel_vars):
            raise ValueError(f"dout_values length {len(dout_values)} must equal 2^{len(sel_vars)}")
        
        self.sel_vars = sel_vars
        self.dout_values = dout_values
        self.root = self._build_tree(sel_vars, dout_values, 0)
        self._simplify_tree()
    
    def _build_tree(self, sel_vars: List[str], values: List[int], depth: int) -> TreeNode:
        """Recursively build the binary decision tree."""
        if depth >= len(sel_vars):
            # Terminal node
            if len(values) != 1:
                raise ValueError(f"Expected single value at terminal, got {len(values)}")
            return TreeNode(value=values[0])
        
        # Internal node
        mid = len(values) // 2
        left_values = values[:mid]  # sel[depth] = 0 (left branch)
        right_values = values[mid:]  # sel[depth] = 1 (right branch)
        
        node = TreeNode(sel_var=sel_vars[depth])
        node.left = self._build_tree(sel_vars, left_values, depth + 1)   # sel=0
        node.right = self._build_tree(sel_vars, right_values, depth + 1)  # sel=1
        
        return node
    
    def _simplify_tree(self):
        """Simplify the tree from terminal nodes to root."""
        self.root = self._simplify_node(self.root)
    
    def _simplify_node(self, node: TreeNode) -> TreeNode:
        """
        Recursively simplify a node according to the rules:
        - Both children are -1 → current node = -1
        - Both children same → current node = same value  
        - Left=1, Right=0 → current node = ~sel[i]
        - Left=-1 → current node = right child
        - Right=-1 → current node = left child
        """
        if node.is_terminal():
            return node
        
        # Recursively simplify children first
        left = self._simplify_node(node.left)
        right = self._simplify_node(node.right)
        
        # Apply simplification rules
        if left.is_terminal() and right.is_terminal():
            left_val = left.value
            right_val = right.value
            
            # Both children are -1 (don't care)
            if left_val == -1 and right_val == -1:
                return TreeNode(value=-1)
            
            # Both children are the same
            if left_val == right_val:
                return TreeNode(value=left_val)
            
            # Left=-1, return right
            if left_val == -1:
                return right
            
            # Right=-1, return left  
            if right_val == -1:
                return left
        
        # Check if one child is don't care
        elif left.is_terminal() and left.value == -1:
            return right
        elif right.is_terminal() and right.value == -1:
            return left
        
        # Check if both subtrees are identical
        elif left == right:
            return left
        
        # Update the node with simplified children
        node.left = left
        node.right = right
        return node
    
    def get_expression(self) -> str:
        """Generate mux_tree expression string."""
        return self._node_to_expression(self.root)
    
    def _node_to_expression(self, node: TreeNode) -> str:
        """Convert node to expression string."""
        if node.is_terminal():
            return self._value_to_string(node.value)
        
        left_expr = self._node_to_expression(node.left)   # sel=0
        right_expr = self._node_to_expression(node.right)  # sel=1
        
        return f"({node.sel_var} ? {right_expr} : {left_expr})"
    
    def _value_to_string(self, value: int) -> str:
        """Convert encoded value to string representation."""
        if value == 1:
            return "1'b1"
        elif value == 0:
            return "1'b0"
        elif value == -1:
            return "1'bx"
        elif value >= 2:
            # din[i] or ~din[i]
            if value % 2 == 0:
                # din[i]: value = 2*i+2, so i = (value-2)/2
                i = (value - 2) // 2
                return f"din[{i}]"
            else:
                # ~din[i]: value = 2*i+3, so i = (value-3)/2
                i = (value - 3) // 2
                return f"~din[{i}]"
        else:
            raise ValueError(f"Invalid encoded value: {value}")
    
    def count_nodes(self, visited: Set[TreeNode] = None) -> int:
        """Count unique nodes in this tree."""
        if visited is None:
            visited = set()
        return self._count_nodes_recursive(self.root, visited)
    
    def _count_nodes_recursive(self, node: TreeNode, visited: Set[TreeNode]) -> int:
        """Recursively count nodes, avoiding duplicates."""
        if node in visited:
            return 0
        
        visited.add(node)
        count = 1
        
        if not node.is_terminal():
            count += self._count_nodes_recursive(node.left, visited)
            count += self._count_nodes_recursive(node.right, visited)
        
        return count


class BinaryDecisionForest:
    """
    Binary Decision Forest for multiple output bits.
    
    Manages multiple BinaryDecisionTree instances and provides optimization
    for minimizing total AIG nodes through sel variable reordering.
    """
    
    def __init__(self, sel_vars: List[str], dout_lists: List[List[int]]):
        """
        Initialize binary decision forest.
        
        Args:
            sel_vars: Selection variables in order (index 0 = highest level)
            dout_lists: List of output value lists, one per dout bit
        """
        self.sel_vars = sel_vars
        self.dout_lists = dout_lists
        self.trees = []
        
        # Build trees for each output bit
        for i, dout_values in enumerate(dout_lists):
            tree = BinaryDecisionTree(sel_vars, dout_values)
            self.trees.append(tree)
    
    def get_expressions(self) -> List[str]:
        """Get mux_tree expressions for all output bits."""
        return [tree.get_expression() for tree in self.trees]
    
    def count_total_nodes(self) -> int:
        """Count total unique AIG nodes across all trees."""
        visited = set()
        total = 0
        for tree in self.trees:
            total += tree.count_nodes(visited)
        return total
    
    def optimize_sel_order(self) -> Tuple[List[str], int]:
        """
        Find the optimal sel variable ordering that minimizes total AIG nodes.
        
        Returns:
            Tuple of (optimal_sel_vars, minimum_node_count)
        """
        min_nodes = float('inf')
        best_order = self.sel_vars.copy()
        current_nodes = self.count_total_nodes()
        min_nodes = current_nodes
        
        print(f"Initial order {self.sel_vars}: {current_nodes} nodes")
        
        # Try all permutations of sel_vars
        for perm in itertools.permutations(self.sel_vars):
            perm_list = list(perm)
            
            if perm_list == self.sel_vars:
                continue  # Skip the original order
            
            # Create forest with this ordering
            try:
                # Need to reorder dout_lists according to new sel variable order
                reordered_dout_lists = self._reorder_dout_lists(perm_list)
                temp_forest = BinaryDecisionForest(perm_list, reordered_dout_lists)
                node_count = temp_forest.count_total_nodes()
                
                print(f"Testing order {perm_list}: {node_count} nodes")
                
                if node_count < min_nodes:
                    min_nodes = node_count
                    best_order = perm_list
                    print(f"  New best: {min_nodes} nodes")
            except Exception as e:
                # Skip invalid permutations
                print(f"  Error with {perm_list}: {e}")
                continue
        
        return best_order, min_nodes
    
    def _reorder_dout_lists(self, new_sel_order: List[str]) -> List[List[int]]:
        """
        Reorder dout_lists according to new sel variable ordering.
        
        The key insight is that we need to map each truth table row (combination)
        from the old variable order to the new variable order.
        """
        num_vars = len(self.sel_vars)
        num_combinations = 2 ** num_vars
        
        # Create mapping from old variable positions to new positions
        var_mapping = {}
        for old_pos, var in enumerate(self.sel_vars):
            new_pos = new_sel_order.index(var)
            var_mapping[old_pos] = new_pos
        
        reordered_lists = []
        for dout_values in self.dout_lists:
            reordered_values = [0] * num_combinations
            
            for old_idx in range(num_combinations):
                # Extract bit values for old index
                old_bits = []
                temp = old_idx
                for _ in range(num_vars):
                    old_bits.append(temp & 1)
                    temp >>= 1
                old_bits.reverse()  # MSB first
                
                # Remap bits according to new variable order
                new_bits = [0] * num_vars
                for old_pos, bit_val in enumerate(old_bits):
                    new_pos = var_mapping[old_pos]
                    new_bits[new_pos] = bit_val
                
                # Calculate new index from remapped bits
                new_idx = 0
                for bit in new_bits:
                    new_idx = (new_idx << 1) | bit
                
                reordered_values[new_idx] = dout_values[old_idx]
            
            reordered_lists.append(reordered_values)
        
        return reordered_lists
    
    def generate_verilog_assigns(self, dout_name: str = "dout") -> List[str]:
        """
        Generate Verilog assign statements for all output bits.
        
        Args:
            dout_name: Name of the output signal
            
        Returns:
            List of assign statements
        """
        expressions = self.get_expressions()
        assigns = []
        
        for i, expr in enumerate(expressions):
            assigns.append(f"assign {dout_name}[{i}] = {expr};")
        
        return assigns
    
    def print_summary(self):
        """Print summary of the forest."""
        print(f"Binary Decision Forest Summary:")
        print(f"  Selection variables: {self.sel_vars}")
        print(f"  Number of output bits: {len(self.trees)}")
        print(f"  Total AIG nodes: {self.count_total_nodes()}")
        print(f"  Expressions:")
        for i, expr in enumerate(self.get_expressions()):
            print(f"    dout[{i}] = {expr}")


def parse_verilog_example():
    """
    Parse the example from the problem statement to demonstrate usage.
    """
    # From the example:
    # casez (sel)
    #     4'b1011 : dout = {1'b1, 1'b1, 1'b0, 1'b1};      // sel=1011 (11)
    #     4'b0101 : dout = {1'b0, 1'b1, ~din[2], 1'b1};    // sel=0101 (5)  
    #     4'b1101 : dout = {din[1], 1'b0, 1'b1, 1'b0};     // sel=1101 (13)
    #     4'b1000 : dout = {1'b0, 1'b1, 1'b0, din[3]};     // sel=1000 (8)
    #     default: dout = {1'bx, 1'bx, 1'bx, 1'bx};
    
    sel_vars = ["sel[3]", "sel[2]", "sel[1]", "sel[0]"]  # 4-bit sel
    
    # Initialize all outputs to don't care (-1)
    dout_lists = [[-1] * 16 for _ in range(4)]  # 4 output bits, 16 combinations
    
    # Set specific cases
    # Case 4'b1011 (binary 1011 = 11): dout = {1, 1, 0, 1}
    dout_lists[3][11] = 1  # dout[3] = 1
    dout_lists[2][11] = 1  # dout[2] = 1  
    dout_lists[1][11] = 0  # dout[1] = 0
    dout_lists[0][11] = 1  # dout[0] = 1
    
    # Case 4'b0101 (binary 0101 = 5): dout = {0, 1, ~din[2], 1}
    dout_lists[3][5] = 0   # dout[3] = 0
    dout_lists[2][5] = 1   # dout[2] = 1
    dout_lists[1][5] = 7   # dout[1] = ~din[2] (2*2+3 = 7)
    dout_lists[0][5] = 1   # dout[0] = 1
    
    # Case 4'b1101 (binary 1101 = 13): dout = {din[1], 0, 1, 0}
    dout_lists[3][13] = 4  # dout[3] = din[1] (2*1+2 = 4)
    dout_lists[2][13] = 0  # dout[2] = 0
    dout_lists[1][13] = 1  # dout[1] = 1
    dout_lists[0][13] = 0  # dout[0] = 0
    
    # Case 4'b1000 (binary 1000 = 8): dout = {0, 1, 0, din[3]}
    dout_lists[3][8] = 0   # dout[3] = 0
    dout_lists[2][8] = 1   # dout[2] = 1
    dout_lists[1][8] = 0   # dout[1] = 0
    dout_lists[0][8] = 8   # dout[0] = din[3] (2*3+2 = 8)
    
    return sel_vars, dout_lists


if __name__ == "__main__":
    # Example usage
    print("Binary Decision Forest Example")
    print("=" * 40)
    
    # Parse the Verilog example
    sel_vars, dout_lists = parse_verilog_example()
    
    # Create forest
    forest = BinaryDecisionForest(sel_vars, dout_lists)
    forest.print_summary()
    
    print("\nVerilog assign statements:")
    assigns = forest.generate_verilog_assigns()
    for assign in assigns:
        print(f"  {assign}")
    
    # Try optimization
    print(f"\nOptimizing sel variable order...")
    optimal_order, min_nodes = forest.optimize_sel_order()
    print(f"Optimal order: {optimal_order}")
    print(f"Minimum nodes: {min_nodes}")