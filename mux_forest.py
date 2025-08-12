"""
Binary Decision Forest Implementation for Verilog pmux to mux_tree conversion.

This module implements a system to convert Verilog pmux (casez statements) to 
mux_tree (nested ternary operators) with optimized AIG node count.
"""

from typing import List, Dict, Set, Tuple, Optional, Union
from dataclasses import dataclass
import itertools
import random
import math


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
    for minimizing total AIG nodes. Each tree can have its own optimal 
    sel variable order for maximum simplification.
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
        self.tree_sel_orders = []  # Track sel order for each tree
        
        # Build trees for each output bit with initial order
        for i, dout_values in enumerate(dout_lists):
            tree = BinaryDecisionTree(sel_vars, dout_values)
            self.trees.append(tree)
            self.tree_sel_orders.append(sel_vars.copy())
    
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
    
    def optimize_sel_order(self, max_iterations: int = 1000, use_heuristics: bool = True, 
                          hybrid_strategy: bool = True) -> Tuple[List[List[str]], int]:
        """
        Find the optimal sel variable ordering to minimize total AIG nodes.
        
        Uses a hybrid strategy that tries both unified (same order for all trees) and 
        independent (different order per tree) optimization, then chooses the better result.
        
        Args:
            max_iterations: Maximum number of iterations for optimization (default: 1000)
            use_heuristics: Whether to use heuristic optimization for large problems (default: True)
            hybrid_strategy: Whether to try both unified and independent approaches (default: True)
        
        Returns:
            Tuple of (optimal_sel_orders_per_tree, minimum_total_node_count)
        """
        num_vars = len(self.sel_vars)
        initial_nodes = self.count_total_nodes()
        
        print(f"Initial configuration: {initial_nodes} total nodes")
        print(f"Strategy: {'Hybrid (unified vs independent)' if hybrid_strategy else 'Independent only'}")
        
        if not hybrid_strategy:
            # Use only independent optimization (original approach)
            return self._optimize_independent(max_iterations, use_heuristics)
        
        # Hybrid strategy: try both unified and independent approaches
        print(f"\n=== TRYING UNIFIED OPTIMIZATION ===")
        unified_orders, unified_nodes = self._optimize_unified(max_iterations, use_heuristics)
        
        print(f"\n=== TRYING INDEPENDENT OPTIMIZATION ===")
        independent_orders, independent_nodes = self._optimize_independent(max_iterations, use_heuristics)
        
        # Choose the better result
        if unified_nodes <= independent_nodes:
            print(f"\n=== UNIFIED APPROACH WINS ===")
            print(f"Unified: {unified_nodes} nodes vs Independent: {independent_nodes} nodes")
            print(f"Using unified order for all trees: {unified_orders[0]}")
            
            # Apply unified order to all trees
            self._apply_unified_order(unified_orders[0])
            return unified_orders, unified_nodes
        else:
            print(f"\n=== INDEPENDENT APPROACH WINS ===")
            print(f"Independent: {independent_nodes} nodes vs Unified: {unified_nodes} nodes")
            print(f"Using independent orders per tree")
            
            # Independent result is already applied, just return it
            return independent_orders, independent_nodes
    
    def _optimize_unified(self, max_iterations: int, use_heuristics: bool) -> Tuple[List[List[str]], int]:
        """
        Optimize using a unified sel order for all trees (maximizes mux sharing).
        
        Returns:
            Tuple of (orders_per_tree, total_node_count) where all orders are the same
        """
        num_vars = len(self.sel_vars)
        initial_nodes = self.count_total_nodes()
        best_order = self.sel_vars.copy()
        best_nodes = initial_nodes
        
        print(f"Optimizing unified order for {len(self.trees)} trees with {num_vars} variables...")
        
        if num_vars <= 6:
            # Exhaustive search for unified order
            print("Using exhaustive search for unified optimization...")
            tested = 0
            for perm in itertools.permutations(self.sel_vars):
                perm_list = list(perm)
                tested += 1
                
                total_nodes = self._evaluate_unified_order(perm_list)
                if total_nodes < best_nodes:
                    best_nodes = total_nodes
                    best_order = perm_list
                    print(f"  New best unified order: {best_nodes} nodes with {best_order}")
            
            print(f"Exhaustive search completed. Tested {tested} permutations.")
        
        elif use_heuristics:
            # Heuristic optimization for unified order
            print("Using heuristic optimization for unified order...")
            
            # Cross-tree variable importance analysis
            var_importance = self._calculate_unified_variable_importance()
            
            # Greedy construction based on cross-tree analysis
            greedy_order, greedy_nodes = self._greedy_construction_unified(var_importance)
            if greedy_nodes < best_nodes:
                best_nodes = greedy_nodes
                best_order = greedy_order
                print(f"  Greedy unified improved to {best_nodes} nodes")
            
            # Local search for unified order
            local_order, local_nodes = self._local_search_unified(best_order, max_iterations // 2)
            if local_nodes < best_nodes:
                best_nodes = local_nodes
                best_order = local_order
                print(f"  Local search unified improved to {best_nodes} nodes")
            
            # Advanced cross-tree optimization (run with more iterations for better results)
            advanced_order, advanced_nodes = self._advanced_unified_optimization(best_order, max_iterations)
            if advanced_nodes < best_nodes:
                best_nodes = advanced_nodes
                best_order = advanced_order
                print(f"  Advanced unified improved to {best_nodes} nodes")
        
        else:
            # Random search for unified order
            print(f"Using random search for unified order with {max_iterations} iterations...")
            for i in range(max_iterations):
                test_order = self.sel_vars.copy()
                random.shuffle(test_order)
                total_nodes = self._evaluate_unified_order(test_order)
                
                if total_nodes < best_nodes:
                    best_nodes = total_nodes
                    best_order = test_order
                    print(f"  Random unified improved to {best_nodes} nodes at iteration {i}")
        
        print(f"Unified optimization complete: {initial_nodes} -> {best_nodes} nodes")
        print(f"Best unified order: {best_order}")
        
        # Return the same order for all trees
        unified_orders = [best_order.copy() for _ in range(len(self.trees))]
        return unified_orders, best_nodes
    
    def _optimize_independent(self, max_iterations: int, use_heuristics: bool) -> Tuple[List[List[str]], int]:
        """
        Optimize each tree independently (original approach).
        
        Returns:
            Tuple of (orders_per_tree, total_node_count)
        """
        num_vars = len(self.sel_vars)
        current_nodes = self.count_total_nodes()
        
        print(f"Optimizing {len(self.trees)} trees independently with {num_vars} variables each...")
        
        optimal_orders = []
        total_improvements = 0
        
        # Optimize each tree independently
        for tree_idx in range(len(self.trees)):
            print(f"\nOptimizing tree {tree_idx}...")
            
            if num_vars <= 6:
                # For small problems, use exhaustive search for this tree
                best_order, best_nodes = self._optimize_single_tree_exhaustive(tree_idx)
            elif use_heuristics:
                # For large problems, use heuristic optimization for this tree
                best_order, best_nodes = self._optimize_single_tree_heuristic(tree_idx, max_iterations)
            else:
                # Limited random search for this tree
                best_order, best_nodes = self._optimize_single_tree_random(tree_idx, max_iterations)
            
            optimal_orders.append(best_order)
            
            # Update this tree with its optimal order
            original_nodes = self.trees[tree_idx].count_nodes()
            reordered_dout_values = self._reorder_single_dout_list(self.dout_lists[tree_idx], best_order)
            self.trees[tree_idx] = BinaryDecisionTree(best_order, reordered_dout_values)
            self.tree_sel_orders[tree_idx] = best_order
            
            new_nodes = self.trees[tree_idx].count_nodes()
            improvement = original_nodes - new_nodes
            total_improvements += improvement
            
            print(f"  Tree {tree_idx}: {original_nodes} -> {new_nodes} nodes (improvement: {improvement})")
        
        final_nodes = self.count_total_nodes()
        print(f"Independent optimization complete:")
        print(f"  Initial total: {current_nodes} nodes")
        print(f"  Final total: {final_nodes} nodes") 
        print(f"  Total improvement: {current_nodes - final_nodes} nodes")
        
        return optimal_orders, final_nodes
    
    def _apply_unified_order(self, unified_order: List[str]):
        """Apply a unified order to all trees."""
        for tree_idx in range(len(self.trees)):
            reordered_dout_values = self._reorder_single_dout_list(self.dout_lists[tree_idx], unified_order)
            self.trees[tree_idx] = BinaryDecisionTree(unified_order, reordered_dout_values)
            self.tree_sel_orders[tree_idx] = unified_order.copy()
    
    def _evaluate_unified_order(self, order: List[str]) -> int:
        """Evaluate total AIG node count for a unified sel variable order."""
        try:
            # Create temporary trees with the new order
            temp_trees = []
            for tree_idx in range(len(self.trees)):
                reordered_dout_values = self._reorder_single_dout_list(self.dout_lists[tree_idx], order)
                temp_tree = BinaryDecisionTree(order, reordered_dout_values)
                temp_trees.append(temp_tree)
            
            # Count total nodes with sharing
            visited = set()
            total = 0
            for tree in temp_trees:
                total += tree.count_nodes(visited)
            
            return total
        except Exception:
            return float('inf')
    
    def _calculate_unified_variable_importance(self) -> Dict[str, float]:
        """Calculate variable importance considering all trees together."""
        importance = {var: 0.0 for var in self.sel_vars}
        
        for var in self.sel_vars:
            total_score = 0.0
            
            # Test variable importance across all trees
            for pos in range(min(3, len(self.sel_vars))):
                test_order = self.sel_vars.copy()
                test_order.remove(var)
                test_order.insert(pos, var)
                
                try:
                    total_nodes = self._evaluate_unified_order(test_order)
                    weight = 1.0 / (pos + 1)
                    total_score += weight / total_nodes if total_nodes > 0 else weight
                except:
                    continue
            
            importance[var] = total_score
        
        return importance
    
    def _greedy_construction_unified(self, var_importance: Dict[str, float]) -> Tuple[List[str], int]:
        """Greedy construction for unified order based on cross-tree importance."""
        sorted_vars = sorted(self.sel_vars, key=lambda v: var_importance.get(v, 0), reverse=True)
        
        try:
            total_nodes = self._evaluate_unified_order(sorted_vars)
            return sorted_vars, total_nodes
        except:
            return self.sel_vars.copy(), self._evaluate_unified_order(self.sel_vars)
    
    def _local_search_unified(self, initial_order: List[str], max_iterations: int) -> Tuple[List[str], int]:
        """Local search optimization for unified order."""
        current_order = initial_order.copy()
        current_nodes = self._evaluate_unified_order(current_order)
        best_order = current_order.copy()
        best_nodes = current_nodes
        
        iterations = 0
        improvements = 0
        
        while iterations < max_iterations:
            iterations += 1
            improved = False
            
            # Try adjacent swaps
            for i in range(len(current_order) - 1):
                test_order = current_order.copy()
                test_order[i], test_order[i + 1] = test_order[i + 1], test_order[i]
                
                test_nodes = self._evaluate_unified_order(test_order)
                
                if test_nodes < current_nodes:
                    current_order = test_order
                    current_nodes = test_nodes
                    improved = True
                    improvements += 1
                    
                    if test_nodes < best_nodes:
                        best_order = test_order.copy()
                        best_nodes = test_nodes
                    break
            
            # Try position moves every 10 iterations
            if not improved and iterations % 10 == 0:
                for i in range(len(current_order)):
                    for j in range(len(current_order)):
                        if i == j:
                            continue
                        
                        test_order = current_order.copy()
                        var = test_order.pop(i)
                        test_order.insert(j, var)
                        
                        test_nodes = self._evaluate_unified_order(test_order)
                        
                        if test_nodes < current_nodes:
                            current_order = test_order
                            current_nodes = test_nodes
                            improved = True
                            improvements += 1
                            
                            if test_nodes < best_nodes:
                                best_order = test_order.copy()
                                best_nodes = test_nodes
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_order, best_nodes
    
    def _advanced_unified_optimization(self, initial_order: List[str], max_iterations: int) -> Tuple[List[str], int]:
        """Advanced optimization techniques for unified order with constant folding and sharing analysis."""
        best_order = initial_order.copy()
        best_nodes = self._evaluate_unified_order(best_order)
        
        # Phase 1: Constant-aware optimization
        constant_aware_order, constant_aware_nodes = self._constant_aware_optimization(initial_order)
        if constant_aware_nodes < best_nodes:
            best_order = constant_aware_order
            best_nodes = constant_aware_nodes
            print(f"    Constant-aware optimization improved to {best_nodes} nodes")
        
        # Phase 2: Sharing-aware reordering
        sharing_aware_order, sharing_aware_nodes = self._sharing_aware_optimization(best_order, max_iterations // 3)
        if sharing_aware_nodes < best_nodes:
            best_order = sharing_aware_order
            best_nodes = sharing_aware_nodes
            print(f"    Sharing-aware optimization improved to {best_nodes} nodes")
        
        # Phase 3: Simulated annealing with adaptive cooling
        annealing_order, annealing_nodes = self._adaptive_simulated_annealing(best_order, max_iterations // 3)
        if annealing_nodes < best_nodes:
            best_order = annealing_order
            best_nodes = annealing_nodes
            print(f"    Adaptive annealing improved to {best_nodes} nodes")
        
        # Phase 4: Multi-level optimization
        multilevel_order, multilevel_nodes = self._multilevel_optimization(best_order, max_iterations // 3)
        if multilevel_nodes < best_nodes:
            best_order = multilevel_order
            best_nodes = multilevel_nodes
            print(f"    Multi-level optimization improved to {best_nodes} nodes")
        
        return best_order, best_nodes
    
    def _constant_aware_optimization(self, initial_order: List[str]) -> Tuple[List[str], int]:
        """Optimize considering constant values (0, 1, x) and their impact on sharing."""
        # Analyze which variables have the most constant/don't-care values
        var_entropy = {}
        
        for var_idx, var in enumerate(self.sel_vars):
            total_entropy = 0.0
            
            for tree_idx in range(len(self.trees)):
                # Calculate entropy for this variable in this tree
                values_0 = []  # Values when var=0
                values_1 = []  # Values when var=1
                
                num_combinations = len(self.dout_lists[tree_idx])
                for combo_idx in range(num_combinations):
                    # Check if this combination has var=0 or var=1
                    bit_val = (combo_idx >> (len(self.sel_vars) - 1 - var_idx)) & 1
                    value = self.dout_lists[tree_idx][combo_idx]
                    
                    if bit_val == 0:
                        values_0.append(value)
                    else:
                        values_1.append(value)
                
                # Calculate entropy - variables with more constants have lower entropy
                def calc_entropy(values):
                    if not values:
                        return 0
                    unique_vals = set(values)
                    entropy = 0
                    for val in unique_vals:
                        prob = values.count(val) / len(values)
                        if prob > 0:
                            entropy -= prob * math.log2(prob)
                    return entropy
                
                entropy_0 = calc_entropy(values_0)
                entropy_1 = calc_entropy(values_1)
                total_entropy += entropy_0 + entropy_1
            
            var_entropy[var] = total_entropy
        
        # Sort variables by entropy (low entropy = high constant ratio = should be deeper)
        sorted_vars = sorted(self.sel_vars, key=lambda v: var_entropy[v], reverse=True)
        
        # Try this constant-aware ordering
        constant_nodes = self._evaluate_unified_order(sorted_vars)
        
        # Try hybrid approach: high-entropy vars first, then low-entropy
        mid_point = len(sorted_vars) // 2
        high_entropy = sorted_vars[:mid_point]
        low_entropy = sorted_vars[mid_point:]
        
        # Try different combinations
        best_order = initial_order.copy()
        best_nodes = self._evaluate_unified_order(best_order)
        
        for order in [sorted_vars, sorted_vars[::-1], high_entropy + low_entropy, low_entropy + high_entropy]:
            nodes = self._evaluate_unified_order(order)
            if nodes < best_nodes:
                best_nodes = nodes
                best_order = order
        
        return best_order, best_nodes
    
    def _sharing_aware_optimization(self, initial_order: List[str], max_iterations: int) -> Tuple[List[str], int]:
        """Optimize considering potential sharing between trees."""
        best_order = initial_order.copy()
        best_nodes = self._evaluate_unified_order(best_order)
        
        # Analyze sharing potential at different levels
        for iteration in range(max_iterations):
            # Try to group variables that create similar subtree patterns
            test_order = self._reorder_for_sharing(initial_order, iteration)
            test_nodes = self._evaluate_unified_order(test_order)
            
            if test_nodes < best_nodes:
                best_nodes = test_nodes
                best_order = test_order
        
        return best_order, best_nodes
    
    def _reorder_for_sharing(self, order: List[str], iteration: int) -> List[str]:
        """Reorder variables to maximize sharing potential."""
        test_order = order.copy()
        
        # Strategy based on iteration
        if iteration % 4 == 0:
            # Move high-impact variables to front
            random.shuffle(test_order[:3])
        elif iteration % 4 == 1:
            # Move middle variables around
            mid = len(test_order) // 2
            random.shuffle(test_order[mid-2:mid+2])
        elif iteration % 4 == 2:
            # Reverse some subsequence
            i = random.randint(0, len(test_order) - 3)
            j = random.randint(i + 2, len(test_order))
            test_order[i:j] = test_order[i:j][::-1]
        else:
            # Random adjacent swap
            i = random.randint(0, len(test_order) - 2)
            test_order[i], test_order[i+1] = test_order[i+1], test_order[i]
        
        return test_order
    
    def _adaptive_simulated_annealing(self, initial_order: List[str], max_iterations: int) -> Tuple[List[str], int]:
        """Simulated annealing with adaptive temperature and cooling schedule."""
        best_order = initial_order.copy()
        best_nodes = self._evaluate_unified_order(best_order)
        
        current_order = best_order.copy()
        current_nodes = best_nodes
        
        # Adaptive parameters
        temperature = max(100.0, best_nodes * 0.1)  # Scale with problem size
        min_temperature = 1.0
        cooling_rate = 0.95
        reheat_threshold = max_iterations // 4
        last_improvement = 0
        
        for iteration in range(max_iterations):
            # Generate neighbor with varied strategies
            test_order = self._generate_neighbor(current_order, iteration, max_iterations)
            test_nodes = self._evaluate_unified_order(test_order)
            
            # Accept or reject based on simulated annealing
            delta = test_nodes - current_nodes
            accept = False
            
            if delta < 0:
                accept = True
            elif temperature > min_temperature:
                probability = math.exp(-delta / temperature)
                accept = random.random() < probability
            
            if accept:
                current_order = test_order
                current_nodes = test_nodes
                
                if test_nodes < best_nodes:
                    best_order = test_order.copy()
                    best_nodes = test_nodes
                    last_improvement = iteration
            
            # Adaptive cooling and reheating
            if iteration - last_improvement > reheat_threshold:
                # Reheat
                temperature = min(temperature * 2.0, best_nodes * 0.1)
                last_improvement = iteration
            else:
                temperature *= cooling_rate
                temperature = max(temperature, min_temperature)
        
        return best_order, best_nodes
    
    def _generate_neighbor(self, order: List[str], iteration: int, max_iterations: int) -> List[str]:
        """Generate neighbor solution with strategy based on iteration phase."""
        test_order = order.copy()
        phase = iteration / max_iterations
        
        if phase < 0.3:  # Early phase: large moves
            # Block move
            block_size = min(3, len(test_order) // 3)
            start = random.randint(0, len(test_order) - block_size)
            block = test_order[start:start + block_size]
            del test_order[start:start + block_size]
            new_pos = random.randint(0, len(test_order))
            test_order[new_pos:new_pos] = block
            
        elif phase < 0.7:  # Middle phase: medium moves
            # Triple swap
            indices = random.sample(range(len(test_order)), min(3, len(test_order)))
            values = [test_order[i] for i in indices]
            random.shuffle(values)
            for i, val in zip(indices, values):
                test_order[i] = val
                
        else:  # Late phase: fine tuning
            # Adjacent swap or single move
            if random.random() < 0.7:
                i = random.randint(0, len(test_order) - 2)
                test_order[i], test_order[i+1] = test_order[i+1], test_order[i]
            else:
                i = random.randint(0, len(test_order) - 1)
                j = random.randint(0, len(test_order) - 1)
                if i != j:
                    var = test_order.pop(i)
                    test_order.insert(j, var)
        
        return test_order
    
    def _multilevel_optimization(self, initial_order: List[str], max_iterations: int) -> Tuple[List[str], int]:
        """Multi-level optimization with divide-and-conquer approach."""
        if len(initial_order) <= 4:
            return initial_order, self._evaluate_unified_order(initial_order)
        
        best_order = initial_order.copy()
        best_nodes = self._evaluate_unified_order(best_order)
        
        # Divide variables into groups and optimize within groups
        group_size = min(6, len(initial_order) // 2)
        
        for start_pos in range(0, len(initial_order) - group_size + 1, max(1, group_size // 2)):
            end_pos = min(start_pos + group_size, len(initial_order))
            
            # Extract group and optimize
            prefix = best_order[:start_pos]
            group = best_order[start_pos:end_pos]
            suffix = best_order[end_pos:]
            
            # Try all permutations of the group (if small enough)
            if len(group) <= 5:
                best_group = group
                best_group_score = float('inf')
                
                for perm in itertools.permutations(group):
                    test_order = prefix + list(perm) + suffix
                    score = self._evaluate_unified_order(test_order)
                    
                    if score < best_group_score:
                        best_group_score = score
                        best_group = list(perm)
                
                test_order = prefix + best_group + suffix
                test_nodes = self._evaluate_unified_order(test_order)
                
                if test_nodes < best_nodes:
                    best_nodes = test_nodes
                    best_order = test_order
        
        return best_order, best_nodes
    
    def optimize_with_constant_folding(self, max_iterations: int = 1000) -> Tuple[List[List[str]], int]:
        """
        Enhanced optimization with constant folding and special value optimization.
        
        This method specifically targets the issues mentioned by the user:
        - Mux gate reuse through unified ordering
        - Special node (1,0,x) utilization 
        - Advanced sel signal reordering
        """
        initial_nodes = self.count_total_nodes()
        print(f"Starting constant folding optimization from {initial_nodes} nodes...")
        
        # Step 1: Apply constant folding to all trees
        original_trees = [tree for tree in self.trees]
        folded_improvements = 0
        
        for tree_idx in range(len(self.trees)):
            original_count = self.trees[tree_idx].count_nodes()
            self.trees[tree_idx] = self._apply_constant_folding(self.trees[tree_idx])
            new_count = self.trees[tree_idx].count_nodes()
            folded_improvements += original_count - new_count
        
        folded_nodes = self.count_total_nodes()
        print(f"Constant folding: {initial_nodes} -> {folded_nodes} nodes (improvement: {folded_improvements})")
        
        # Step 2: Run hybrid optimization on folded trees
        optimal_orders, optimized_nodes = self.optimize_sel_order(max_iterations, True, True)
        
        final_improvement = initial_nodes - optimized_nodes
        print(f"Total optimization: {initial_nodes} -> {optimized_nodes} nodes (improvement: {final_improvement})")
        
        return optimal_orders, optimized_nodes
    
    def _apply_constant_folding(self, tree: 'BinaryDecisionTree') -> 'BinaryDecisionTree':
        """Apply constant folding optimizations to a tree."""
        # Create a new optimized tree by folding constants
        folded_root = self._fold_constants_recursive(tree.root)
        
        # Create new tree with folded structure
        new_tree = BinaryDecisionTree(tree.sel_vars, [])
        new_tree.root = folded_root
        
        return new_tree
    
    def _fold_constants_recursive(self, node: TreeNode) -> TreeNode:
        """Recursively apply constant folding optimizations."""
        if node.is_terminal():
            return node
        
        # Recursively fold children
        left_folded = self._fold_constants_recursive(node.left)
        right_folded = self._fold_constants_recursive(node.right)
        
        # Check for folding opportunities
        if left_folded.is_terminal() and right_folded.is_terminal():
            # Both children are terminals
            if left_folded.value == right_folded.value:
                # Same value on both sides - can eliminate this mux
                return TreeNode(value=left_folded.value)
            
            # Check for special patterns
            if left_folded.value == 0 and right_folded.value == 1:
                # sel ? 1 : 0 = sel
                return TreeNode(value=f"sel_var_{node.sel_var}")
            elif left_folded.value == 1 and right_folded.value == 0:
                # sel ? 0 : 1 = ~sel
                return TreeNode(value=f"~sel_var_{node.sel_var}")
        
        # Check if one side is don't care
        if left_folded.is_terminal() and left_folded.value == -1:
            # Don't care on left, use right
            return right_folded
        elif right_folded.is_terminal() and right_folded.value == -1:
            # Don't care on right, use left
            return left_folded
        
        # No folding possible, create new node with folded children
        return TreeNode(sel_var=node.sel_var, left=left_folded, right=right_folded)
    
    def _optimize_single_tree_exhaustive(self, tree_idx: int) -> Tuple[List[str], int]:
        original_nodes = original_tree.count_nodes()
        best_order = self.tree_sel_orders[tree_idx].copy()
        best_nodes = original_nodes
        
        print(f"  Using exhaustive search for tree {tree_idx}...")
        
        tested = 0
        for perm in itertools.permutations(self.sel_vars):
            perm_list = list(perm)
            tested += 1
            
            if perm_list == self.tree_sel_orders[tree_idx]:
                continue  # Skip the current order
            
            try:
                reordered_dout_values = self._reorder_single_dout_list(self.dout_lists[tree_idx], perm_list)
                temp_tree = BinaryDecisionTree(perm_list, reordered_dout_values)
                node_count = temp_tree.count_nodes()
                
                if node_count < best_nodes:
                    best_nodes = node_count
                    best_order = perm_list
                    print(f"    New best for tree {tree_idx}: {best_nodes} nodes with order {best_order}")
            except Exception:
                continue
        
        print(f"  Exhaustive search completed for tree {tree_idx}. Tested {tested} permutations.")
        return best_order, best_nodes
    
    def _optimize_single_tree_heuristic(self, tree_idx: int, max_iterations: int) -> Tuple[List[str], int]:
        """Heuristic optimization for a single tree (large problems)."""
        original_tree = self.trees[tree_idx]
        original_nodes = original_tree.count_nodes()
        best_order = self.tree_sel_orders[tree_idx].copy()
        best_nodes = original_nodes
        
        print(f"  Using heuristic optimization for tree {tree_idx}...")
        
        # Variable importance analysis for this specific tree
        var_importance = self._calculate_single_tree_variable_importance(tree_idx)
        
        # Greedy construction
        greedy_order, greedy_nodes = self._greedy_construction_single_tree(tree_idx, var_importance)
        if greedy_nodes < best_nodes:
            best_nodes = greedy_nodes
            best_order = greedy_order
            print(f"    Greedy improved tree {tree_idx} to {best_nodes} nodes")
        
        # Local search
        local_order, local_nodes = self._local_search_single_tree(tree_idx, best_order, max_iterations // 2)
        if local_nodes < best_nodes:
            best_nodes = local_nodes
            best_order = local_order
            print(f"    Local search improved tree {tree_idx} to {best_nodes} nodes")
        
        # Random restart
        restart_order, restart_nodes = self._random_restart_single_tree(tree_idx, best_order, var_importance, max_iterations // 2)
        if restart_nodes < best_nodes:
            best_nodes = restart_nodes
            best_order = restart_order
            print(f"    Random restart improved tree {tree_idx} to {best_nodes} nodes")
        
        return best_order, best_nodes
    
    def _optimize_single_tree_random(self, tree_idx: int, max_iterations: int) -> Tuple[List[str], int]:
        """Limited random search for a single tree."""
        import random
        
        original_tree = self.trees[tree_idx]
        original_nodes = original_tree.count_nodes()
        best_order = self.tree_sel_orders[tree_idx].copy()
        best_nodes = original_nodes
        
        print(f"  Using random search for tree {tree_idx} with {max_iterations} iterations...")
        
        for i in range(max_iterations):
            test_order = self.sel_vars.copy()
            random.shuffle(test_order)
            
            try:
                reordered_dout_values = self._reorder_single_dout_list(self.dout_lists[tree_idx], test_order)
                temp_tree = BinaryDecisionTree(test_order, reordered_dout_values)
                node_count = temp_tree.count_nodes()
                
                if node_count < best_nodes:
                    best_nodes = node_count
                    best_order = test_order
                    print(f"    New best for tree {tree_idx}: {best_nodes} nodes")
            except Exception:
                continue
        
        return best_order, best_nodes
    def _calculate_single_tree_variable_importance(self, tree_idx: int) -> Dict[str, float]:
        """Calculate importance score for each variable for a specific tree."""
        importance = {}
        
        for var in self.sel_vars:
            score = 0.0
            test_positions = min(3, len(self.sel_vars))
            
            for pos in range(test_positions):
                test_order = self.sel_vars.copy()
                test_order.remove(var)
                test_order.insert(pos, var)
                
                try:
                    reordered_dout_values = self._reorder_single_dout_list(self.dout_lists[tree_idx], test_order)
                    temp_tree = BinaryDecisionTree(test_order, reordered_dout_values)
                    nodes = temp_tree.count_nodes()
                    
                    weight = 1.0 / (pos + 1)
                    score += weight / nodes if nodes > 0 else weight
                except:
                    continue
            
            importance[var] = score
        
        return importance
    
    def _greedy_construction_single_tree(self, tree_idx: int, var_importance: Dict[str, float]) -> Tuple[List[str], int]:
        """Greedy construction for a single tree."""
        sorted_vars = sorted(self.sel_vars, key=lambda v: var_importance.get(v, 0), reverse=True)
        
        try:
            reordered_dout_values = self._reorder_single_dout_list(self.dout_lists[tree_idx], sorted_vars)
            temp_tree = BinaryDecisionTree(sorted_vars, reordered_dout_values)
            nodes = temp_tree.count_nodes()
            return sorted_vars, nodes
        except:
            return self.tree_sel_orders[tree_idx].copy(), self.trees[tree_idx].count_nodes()
    
    def _local_search_single_tree(self, tree_idx: int, initial_order: List[str], max_iterations: int) -> Tuple[List[str], int]:
        """Local search optimization for a single tree."""
        current_order = initial_order.copy()
        current_nodes = self._evaluate_single_tree_order(tree_idx, current_order)
        best_order = current_order.copy()
        best_nodes = current_nodes
        
        iterations = 0
        improvements = 0
        
        while iterations < max_iterations:
            iterations += 1
            improved = False
            
            # Try adjacent swaps
            for i in range(len(current_order) - 1):
                test_order = current_order.copy()
                test_order[i], test_order[i + 1] = test_order[i + 1], test_order[i]
                
                test_nodes = self._evaluate_single_tree_order(tree_idx, test_order)
                
                if test_nodes < current_nodes:
                    current_order = test_order
                    current_nodes = test_nodes
                    improved = True
                    improvements += 1
                    
                    if test_nodes < best_nodes:
                        best_order = test_order.copy()
                        best_nodes = test_nodes
                    break
            
            # Try position moves every 10 iterations
            if not improved and iterations % 10 == 0:
                for i in range(len(current_order)):
                    for j in range(len(current_order)):
                        if i == j:
                            continue
                        
                        test_order = current_order.copy()
                        var = test_order.pop(i)
                        test_order.insert(j, var)
                        
                        test_nodes = self._evaluate_single_tree_order(tree_idx, test_order)
                        
                        if test_nodes < current_nodes:
                            current_order = test_order
                            current_nodes = test_nodes
                            improved = True
                            improvements += 1
                            
                            if test_nodes < best_nodes:
                                best_order = test_order.copy()
                                best_nodes = test_nodes
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_order, best_nodes
    
    def _random_restart_single_tree(self, tree_idx: int, initial_order: List[str], 
                                   var_importance: Dict[str, float], max_iterations: int) -> Tuple[List[str], int]:
        """Random restart search for a single tree."""
        best_order = initial_order.copy()
        best_nodes = self._evaluate_single_tree_order(tree_idx, best_order)
        
        restart_attempts = min(max_iterations // 20, 20)
        
        for restart in range(restart_attempts):
            random_order = self._generate_biased_random_order(var_importance)
            local_order, local_nodes = self._local_search_single_tree(tree_idx, random_order, 10)
            
            if local_nodes < best_nodes:
                best_order = local_order
                best_nodes = local_nodes
        
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
    
    def _evaluate_single_tree_order(self, tree_idx: int, order: List[str]) -> int:
        """Evaluate the AIG node count for a single tree with given variable order."""
        try:
            reordered_dout_values = self._reorder_single_dout_list(self.dout_lists[tree_idx], order)
            temp_tree = BinaryDecisionTree(order, reordered_dout_values)
            return temp_tree.count_nodes()
        except Exception:
            return float('inf')
    
    def _reorder_single_dout_list(self, dout_values: List[int], new_sel_order: List[str]) -> List[int]:
        """Reorder a single dout list according to new sel variable ordering."""
        num_vars = len(self.sel_vars)
        num_combinations = 2 ** num_vars
        
        # Create mapping from old variable positions to new positions
        var_mapping = {}
        for old_pos, var in enumerate(self.sel_vars):
            new_pos = new_sel_order.index(var)
            var_mapping[old_pos] = new_pos
        
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
        
        return reordered_values
    
    def _reorder_dout_lists(self, new_sel_order: List[str]) -> List[List[int]]:
        """
        Reorder dout_lists according to new sel variable ordering.
        
        The key insight is that we need to map each truth table row (combination)
        from the old variable order to the new variable order.
        
        This method is kept for backward compatibility with tests.
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
    
    
    def generate_verilog_assigns(self, dout_name: str = "dout") -> List[str]:
        """
        Generate Verilog assign statements for all output bits.
        Each tree may use its own optimal sel variable order.
        
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
        print(f"  Original selection variables: {self.sel_vars}")
        print(f"  Number of output bits: {len(self.trees)}")
        print(f"  Total AIG nodes: {self.count_total_nodes()}")
        print(f"  Tree-specific orders:")
        for i, order in enumerate(self.tree_sel_orders):
            nodes = self.trees[i].count_nodes()
            print(f"    Tree {i}: {order} ({nodes} nodes)")
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
    print(f"\nOptimizing sel variable order for each tree independently...")
    optimal_orders, min_nodes = forest.optimize_sel_order(max_iterations=1000)
    print(f"Optimal orders per tree: {optimal_orders}")
    print(f"Minimum total nodes: {min_nodes}")
    
    # Print optimized forest summary
    print(f"\nOptimized forest summary:")
    forest.print_summary()
    
    print(f"\nOptimized Verilog assign statements:")
    optimized_assigns = forest.generate_verilog_assigns()
    for assign in optimized_assigns:
        print(f"  {assign}")