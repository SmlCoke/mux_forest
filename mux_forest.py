"""
Binary Decision Forest Implementation for Verilog pmux to mux_tree conversion.

This module implements a system to convert Verilog pmux (casez statements) to 
mux_tree (nested ternary operators) with optimized AIG node count.
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from dataclasses import dataclass
import itertools
import random


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
    
    def optimize_sel_order(self, max_iterations: int = 1000, use_heuristics: bool = True) -> Tuple[List[str], int]:
        """
        Find the optimal sel variable ordering that minimizes total AIG nodes.
        Uses efficient heuristic algorithms for large-scale optimization.
        
        Args:
            max_iterations: Maximum number of permutations to test (default: 1000)
            use_heuristics: Whether to use heuristic optimization for large problems (default: True)
        
        Returns:
            Tuple of (optimal_sel_vars, minimum_node_count)
        """
        num_vars = len(self.sel_vars)
        current_nodes = self.count_total_nodes()
        min_nodes = current_nodes
        best_order = self.sel_vars.copy()
        
        print(f"Initial order {self.sel_vars}: {current_nodes} nodes")
        print(f"Optimizing {num_vars} variables...")
        
        # For small problems (≤6 variables), use exhaustive search
        if num_vars <= 6:
            return self._exhaustive_search()
        
        # For large problems, use heuristic approaches
        if use_heuristics:
            return self._heuristic_optimization(max_iterations)
        else:
            # Limited random sampling
            return self._limited_random_search(max_iterations)
    
    def _exhaustive_search(self) -> Tuple[List[str], int]:
        """Exhaustive search for small problems (≤6 variables)."""
        min_nodes = self.count_total_nodes()
        best_order = self.sel_vars.copy()
        
        print("Using exhaustive search for small problem...")
        
        tested = 0
        for perm in itertools.permutations(self.sel_vars):
            perm_list = list(perm)
            tested += 1
            
            if perm_list == self.sel_vars:
                continue  # Skip the original order
            
            try:
                reordered_dout_lists = self._reorder_dout_lists(perm_list)
                temp_forest = BinaryDecisionForest(perm_list, reordered_dout_lists)
                node_count = temp_forest.count_total_nodes()
                
                if tested % 10 == 0:
                    print(f"  Tested {tested} permutations, current best: {min_nodes}")
                
                if node_count < min_nodes:
                    min_nodes = node_count
                    best_order = perm_list
                    print(f"  New best: {min_nodes} nodes with order {best_order}")
            except Exception as e:
                continue
        
        print(f"Exhaustive search completed. Tested {tested} permutations.")
        return best_order, min_nodes
    
    def _heuristic_optimization(self, max_iterations: int) -> Tuple[List[str], int]:
        """
        Heuristic optimization using greedy local search and variable importance analysis.
        """
        min_nodes = self.count_total_nodes()
        best_order = self.sel_vars.copy()
        
        print("Using heuristic optimization for large problem...")
        
        # Step 1: Variable importance analysis
        var_importance = self._calculate_variable_importance()
        print(f"Variable importance scores: {var_importance}")
        
        # Step 2: Greedy construction based on importance
        greedy_order, greedy_nodes = self._greedy_construction(var_importance)
        if greedy_nodes < min_nodes:
            min_nodes = greedy_nodes
            best_order = greedy_order
            print(f"Greedy construction improved to {min_nodes} nodes")
        
        # Step 3: Local search optimization
        local_order, local_nodes = self._local_search_optimization(best_order, max_iterations // 2)
        if local_nodes < min_nodes:
            min_nodes = local_nodes
            best_order = local_order
            print(f"Local search improved to {min_nodes} nodes")
        
        # Step 4: Random restart with best positions
        restart_order, restart_nodes = self._random_restart_search(best_order, var_importance, max_iterations // 2)
        if restart_nodes < min_nodes:
            min_nodes = restart_nodes
            best_order = restart_order
            print(f"Random restart improved to {min_nodes} nodes")
        
        return best_order, min_nodes
    
    def _calculate_variable_importance(self) -> Dict[str, float]:
        """
        Calculate importance score for each variable based on how much it affects tree structure.
        Higher scores indicate variables that should be placed higher in the tree.
        """
        importance = {}
        
        for var in self.sel_vars:
            # Try placing this variable at different positions and measure impact
            score = 0.0
            test_positions = min(3, len(self.sel_vars))  # Test top 3 positions
            
            for pos in range(test_positions):
                test_order = self.sel_vars.copy()
                test_order.remove(var)
                test_order.insert(pos, var)
                
                try:
                    reordered_dout_lists = self._reorder_dout_lists(test_order)
                    temp_forest = BinaryDecisionForest(test_order, reordered_dout_lists)
                    nodes = temp_forest.count_total_nodes()
                    
                    # Weight higher positions more (position 0 has highest weight)
                    weight = 1.0 / (pos + 1)
                    score += weight / nodes if nodes > 0 else weight
                except:
                    continue
            
            importance[var] = score
        
        return importance
    
    def _greedy_construction(self, var_importance: Dict[str, float]) -> Tuple[List[str], int]:
        """
        Construct variable order greedily based on importance scores.
        """
        # Sort variables by importance (highest first)
        sorted_vars = sorted(self.sel_vars, key=lambda v: var_importance.get(v, 0), reverse=True)
        
        try:
            reordered_dout_lists = self._reorder_dout_lists(sorted_vars)
            temp_forest = BinaryDecisionForest(sorted_vars, reordered_dout_lists)
            nodes = temp_forest.count_total_nodes()
            print(f"Greedy construction order {sorted_vars}: {nodes} nodes")
            return sorted_vars, nodes
        except:
            return self.sel_vars.copy(), self.count_total_nodes()
    
    def _local_search_optimization(self, initial_order: List[str], max_iterations: int) -> Tuple[List[str], int]:
        """
        Local search optimization using adjacent swaps and small moves.
        """
        current_order = initial_order.copy()
        current_nodes = self._evaluate_order(current_order)
        best_order = current_order.copy()
        best_nodes = current_nodes
        
        print(f"Starting local search from {current_nodes} nodes...")
        
        iterations = 0
        improvements = 0
        
        while iterations < max_iterations:
            iterations += 1
            improved = False
            
            # Try adjacent swaps
            for i in range(len(current_order) - 1):
                test_order = current_order.copy()
                test_order[i], test_order[i + 1] = test_order[i + 1], test_order[i]
                
                test_nodes = self._evaluate_order(test_order)
                
                if test_nodes < current_nodes:
                    current_order = test_order
                    current_nodes = test_nodes
                    improved = True
                    improvements += 1
                    
                    if test_nodes < best_nodes:
                        best_order = test_order.copy()
                        best_nodes = test_nodes
                        print(f"  Local search improvement: {best_nodes} nodes")
                    break
            
            # Try moving variables to different positions
            if not improved and iterations % 10 == 0:
                for i in range(len(current_order)):
                    for j in range(len(current_order)):
                        if i == j:
                            continue
                        
                        test_order = current_order.copy()
                        var = test_order.pop(i)
                        test_order.insert(j, var)
                        
                        test_nodes = self._evaluate_order(test_order)
                        
                        if test_nodes < current_nodes:
                            current_order = test_order
                            current_nodes = test_nodes
                            improved = True
                            improvements += 1
                            
                            if test_nodes < best_nodes:
                                best_order = test_order.copy()
                                best_nodes = test_nodes
                                print(f"  Local search improvement: {best_nodes} nodes")
                            break
                    if improved:
                        break
            
            if not improved:
                # No improvement found, stop early
                break
        
        print(f"Local search completed: {iterations} iterations, {improvements} improvements")
        return best_order, best_nodes
    
    def _random_restart_search(self, initial_order: List[str], var_importance: Dict[str, float], 
                              max_iterations: int) -> Tuple[List[str], int]:
        """
        Random restart search with bias towards important variables.
        """
        best_order = initial_order.copy()
        best_nodes = self._evaluate_order(best_order)
        
        print(f"Starting random restart search...")
        
        restart_attempts = min(max_iterations // 20, 50)  # At most 50 restarts
        
        for restart in range(restart_attempts):
            # Generate biased random order
            random_order = self._generate_biased_random_order(var_importance)
            
            # Short local search from this starting point
            local_order, local_nodes = self._local_search_optimization(random_order, 20)
            
            if local_nodes < best_nodes:
                best_order = local_order
                best_nodes = local_nodes
                print(f"  Random restart improvement: {best_nodes} nodes")
        
        return best_order, best_nodes
    
    def _generate_biased_random_order(self, var_importance: Dict[str, float]) -> List[str]:
        """
        Generate random order biased towards important variables being early.
        """
        import random
        
        # Create weighted list favoring important variables
        vars_with_weights = [(var, var_importance.get(var, 0.1)) for var in self.sel_vars]
        
        order = []
        remaining_vars = vars_with_weights.copy()
        
        while remaining_vars:
            # Select variable with probability proportional to importance
            total_weight = sum(weight for _, weight in remaining_vars)
            if total_weight <= 0:
                # Fallback to uniform selection
                selected_var, _ = random.choice(remaining_vars)
            else:
                r = random.random() * total_weight
                cumulative = 0
                selected_var = remaining_vars[0][0]
                for var, weight in remaining_vars:
                    cumulative += weight
                    if r <= cumulative:
                        selected_var = var
                        break
            
            order.append(selected_var)
            remaining_vars = [(v, w) for v, w in remaining_vars if v != selected_var]
        
        return order
    
    def _limited_random_search(self, max_iterations: int) -> Tuple[List[str], int]:
        """
        Limited random search for cases where heuristics are disabled.
        """
        import random
        
        min_nodes = self.count_total_nodes()
        best_order = self.sel_vars.copy()
        
        print(f"Using limited random search with {max_iterations} iterations...")
        
        for i in range(max_iterations):
            # Generate random permutation
            test_order = self.sel_vars.copy()
            random.shuffle(test_order)
            
            try:
                reordered_dout_lists = self._reorder_dout_lists(test_order)
                temp_forest = BinaryDecisionForest(test_order, reordered_dout_lists)
                node_count = temp_forest.count_total_nodes()
                
                if i % 100 == 0:
                    print(f"  Tested {i} random orders, current best: {min_nodes}")
                
                if node_count < min_nodes:
                    min_nodes = node_count
                    best_order = test_order
                    print(f"  New best: {min_nodes} nodes")
            except Exception:
                continue
        
        return best_order, min_nodes
    
    def _evaluate_order(self, order: List[str]) -> int:
        """
        Evaluate the AIG node count for a given variable order.
        Returns a large number if evaluation fails.
        """
        try:
            reordered_dout_lists = self._reorder_dout_lists(order)
            temp_forest = BinaryDecisionForest(order, reordered_dout_lists)
            return temp_forest.count_total_nodes()
        except Exception:
            return float('inf')
    
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
    optimal_order, min_nodes = forest.optimize_sel_order(max_iterations=1000)
    print(f"Optimal order: {optimal_order}")
    print(f"Minimum nodes: {min_nodes}")
    
    # Create optimized forest if order changed
    if optimal_order != sel_vars:
        print(f"\nOptimized forest with reordered variables:")
        optimized_forest = BinaryDecisionForest(optimal_order, 
            forest._reorder_dout_lists(optimal_order))
        optimized_assigns = optimized_forest.generate_verilog_assigns()
        for assign in optimized_assigns:
            print(f"  {assign}")