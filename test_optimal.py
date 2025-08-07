#!/usr/bin/env python3
"""
Test the optimal ordering found by optimization.
"""

from mux_forest import BinaryDecisionForest, parse_verilog_example

def test_optimal_ordering():
    """Test the optimal ordering found."""
    print("Testing optimal sel ordering")
    print("=" * 40)
    
    sel_vars, dout_lists = parse_verilog_example()
    
    # Original ordering
    original_forest = BinaryDecisionForest(sel_vars, dout_lists)
    original_nodes = original_forest.count_total_nodes()
    print(f"Original order {sel_vars}: {original_nodes} nodes")
    
    # Best ordering found: sel[2] first
    best_order = ["sel[2]", "sel[3]", "sel[1]", "sel[0]"]
    reordered_dout_lists = original_forest._reorder_dout_lists(best_order)
    optimal_forest = BinaryDecisionForest(best_order, reordered_dout_lists)
    optimal_nodes = optimal_forest.count_total_nodes()
    
    print(f"Optimal order {best_order}: {optimal_nodes} nodes")
    print(f"Improvement: {original_nodes - optimal_nodes} nodes saved")
    
    print("\nOptimal expressions:")
    expressions = optimal_forest.get_expressions()
    assigns = optimal_forest.generate_verilog_assigns("dout")
    
    for assign in assigns:
        print(f"  {assign}")
    
    print("\nCompare with expected output from problem statement:")
    expected = [
        "assign dout[3] = (sel[2] ? (sel[3] ? 1'b1 : 1'b0) : (sel[0] ? 1'b1 : 1'b0));",
        "assign dout[2] = (sel[2] ? (sel[3] ? 1'b0 : 1'b1) : (sel[0] ? 1'b1 : 1'b1));",
        "assign dout[1] = (sel[2] ? (sel[3] ? 1'b1 : 1'b0) : (sel[0] ? 1'b0 : 1'b0));", 
        "assign dout[0] = (sel[1] ? 1'b1 : (sel[3] ? (sel[0] ? 1'b0 : 1'b0) : 1'b1));",
    ]
    
    for exp in expected:
        print(f"  {exp}")

if __name__ == "__main__":
    test_optimal_ordering()