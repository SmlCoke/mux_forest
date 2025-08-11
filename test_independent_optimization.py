#!/usr/bin/env python3
"""
Test script to validate independent per-tree optimization.

This test demonstrates that each tree can now find its own optimal 
sel variable order, potentially achieving better overall optimization 
than forcing all trees to use the same order.
"""

import sys
from mux_forest import BinaryDecisionForest

def create_test_case_with_different_optimal_orders():
    """
    Create a test case where different trees would benefit from different sel orders.
    """
    # Create a 3-variable case where different trees have different patterns
    sel_vars = ["sel[2]", "sel[1]", "sel[0]"]
    
    # Tree 0: Benefits from sel[0] being at the root (high impact)
    # Pattern: most decisions depend on sel[0]
    dout0 = [
        0,  # 000 -> 0
        1,  # 001 -> 1  
        0,  # 010 -> 0
        1,  # 011 -> 1
        0,  # 100 -> 0  
        1,  # 101 -> 1
        0,  # 110 -> 0
        1,  # 111 -> 1
    ]
    
    # Tree 1: Benefits from sel[2] being at the root
    # Pattern: most decisions depend on sel[2]
    dout1 = [
        0,  # 000 -> 0
        0,  # 001 -> 0
        0,  # 010 -> 0  
        0,  # 011 -> 0
        1,  # 100 -> 1
        1,  # 101 -> 1
        1,  # 110 -> 1
        1,  # 111 -> 1
    ]
    
    # Tree 2: Benefits from sel[1] being at the root
    # Pattern: most decisions depend on sel[1]
    dout2 = [
        0,  # 000 -> 0
        0,  # 001 -> 0
        1,  # 010 -> 1
        1,  # 011 -> 1
        0,  # 100 -> 0
        0,  # 101 -> 0  
        1,  # 110 -> 1
        1,  # 111 -> 1
    ]
    
    dout_lists = [dout0, dout1, dout2]
    return sel_vars, dout_lists

def test_independent_optimization():
    """Test that trees can find different optimal orders independently."""
    print("=" * 60)
    print("Testing Independent Per-Tree Optimization")
    print("=" * 60)
    
    # Create test case
    sel_vars, dout_lists = create_test_case_with_different_optimal_orders()
    
    print(f"Initial sel variables: {sel_vars}")
    print(f"Number of trees: {len(dout_lists)}")
    
    # Create forest
    forest = BinaryDecisionForest(sel_vars, dout_lists)
    
    print(f"\nInitial configuration:")
    forest.print_summary()
    
    # Run optimization  
    print(f"\nRunning independent optimization...")
    optimal_orders, min_nodes = forest.optimize_sel_order(max_iterations=100)
    
    print(f"\nOptimization results:")
    print(f"  Final total nodes: {min_nodes}")
    print(f"  Optimal orders per tree:")
    for i, order in enumerate(optimal_orders):
        print(f"    Tree {i}: {order}")
    
    print(f"\nFinal configuration:")
    forest.print_summary()
    
    # Verify each tree has potentially different orders
    unique_orders = set(tuple(order) for order in optimal_orders)
    print(f"\nUnique orders found: {len(unique_orders)}")
    if len(unique_orders) > 1:
        print("✓ Different trees found different optimal orders!")
    else:
        print("- All trees use the same order (may be optimal for this case)")
    
    # Generate Verilog
    print(f"\nGenerated Verilog:")
    assigns = forest.generate_verilog_assigns()
    for assign in assigns:
        print(f"  {assign}")

def create_asymmetric_case():
    """Create a case where trees have very different terminal patterns."""
    sel_vars = ["sel[1]", "sel[0]"]
    
    # Tree 0: Heavily depends on sel[0]
    dout0 = [0, 1, 0, 1]  # alternating based on sel[0]
    
    # Tree 1: Heavily depends on sel[1] 
    dout1 = [0, 0, 1, 1]  # depends on sel[1]
    
    # Tree 2: Complex pattern
    dout2 = [1, 0, 0, 1]  # XOR pattern
    
    return sel_vars, [dout0, dout1, dout2]

def test_asymmetric_optimization():
    """Test optimization on asymmetric patterns."""
    print("\n" + "=" * 60)
    print("Testing Asymmetric Pattern Optimization")
    print("=" * 60)
    
    sel_vars, dout_lists = create_asymmetric_case()
    
    print(f"Initial sel variables: {sel_vars}")
    
    # Create forest
    forest = BinaryDecisionForest(sel_vars, dout_lists)
    
    print(f"\nBefore optimization:")
    forest.print_summary()
    
    # Run optimization
    optimal_orders, min_nodes = forest.optimize_sel_order(max_iterations=50)
    
    print(f"\nAfter optimization:")
    forest.print_summary()
    
    print(f"\nOptimal orders:")
    for i, order in enumerate(optimal_orders):
        print(f"  Tree {i}: {order}")

if __name__ == "__main__":
    test_independent_optimization()
    test_asymmetric_optimization()
    
    print("\n" + "=" * 60)
    print("Independent optimization tests completed!")
    print("=" * 60)