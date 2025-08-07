#!/usr/bin/env python3
"""
Test optimization algorithm for Binary Decision Forest.
"""

from mux_forest import BinaryDecisionForest, parse_verilog_example
import itertools

def test_optimization_detailed():
    """Test optimization with detailed output."""
    print("Testing optimization in detail")
    print("=" * 40)
    
    sel_vars, dout_lists = parse_verilog_example()
    
    # Test a few specific orderings manually
    orderings_to_test = [
        ["sel[3]", "sel[2]", "sel[1]", "sel[0]"],  # original
        ["sel[2]", "sel[3]", "sel[1]", "sel[0]"],  # swap first two
        ["sel[1]", "sel[2]", "sel[3]", "sel[0]"],  # different order
        ["sel[0]", "sel[1]", "sel[2]", "sel[3]"],  # reverse
    ]
    
    for order in orderings_to_test:
        try:
            print(f"\nTesting order: {order}")
            forest = BinaryDecisionForest(order, dout_lists)
            nodes = forest.count_total_nodes()
            print(f"  Total nodes: {nodes}")
            
            # Show some expressions
            expressions = forest.get_expressions()
            for i, expr in enumerate(expressions[:2]):  # Show first 2
                print(f"  dout[{i}] = {expr}")
        except Exception as e:
            print(f"  Error: {e}")

def test_reordering_logic():
    """Test the dout_lists reordering logic."""
    print("\nTesting reordering logic")
    print("=" * 30)
    
    # Simple 2-bit case for easier debugging
    sel_vars = ["sel[1]", "sel[0]"]
    dout_lists = [[0, 1, 2, 3]]  # Simple increasing pattern
    
    forest = BinaryDecisionForest(sel_vars, dout_lists)
    print(f"Original order {sel_vars}: {dout_lists[0]}")
    print(f"Expression: {forest.get_expressions()[0]}")
    
    # Test reordering to ["sel[0]", "sel[1]"]
    reordered = forest._reorder_dout_lists(["sel[0]", "sel[1]"])
    print(f"Reordered to ['sel[0]', 'sel[1]']: {reordered[0]}")
    
    # Create forest with reordered data
    forest2 = BinaryDecisionForest(["sel[0]", "sel[1]"], reordered)
    print(f"Expression: {forest2.get_expressions()[0]}")

if __name__ == "__main__":
    test_reordering_logic()
    test_optimization_detailed()