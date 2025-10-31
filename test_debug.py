#!/usr/bin/env python3
"""
Test and debug script for the Binary Decision Forest implementation.
"""

from mux_forest import BinaryDecisionForest, BinaryDecisionTree, parse_verilog_example

def test_simple_case():
    """Test with a simple 2-bit sel case."""
    print("Testing simple 2-bit case")
    print("=" * 30)
    
    # Simple case: sel[1:0], 4 combinations
    sel_vars = ["sel[1]", "sel[0]"]
    
    # dout[0]: 00->0, 01->1, 10->din[0], 11->1  
    dout_list0 = [0, 1, 2, 1]  # din[0] = 2*0+2 = 2
    
    tree = BinaryDecisionTree(sel_vars, dout_list0)
    print(f"Expression: {tree.get_expression()}")
    print(f"Nodes: {tree.count_nodes()}")
    print()

def debug_verilog_example():
    """Debug the Verilog example step by step."""
    print("Debugging Verilog example")
    print("=" * 30)
    
    # Let's manually trace the cases
    print("Case analysis:")
    cases = [
        ("4'b1011", [1, 0, 1, 1], [1, 1, 0, 1]),
        ("4'b0101", [0, 1, 0, 1], [0, 1, "~din[2]", 1]),
        ("4'b1101", [1, 1, 0, 1], ["din[1]", 0, 1, 0]),
        ("4'b1000", [1, 0, 0, 0], [0, 1, 0, "din[3]"]),
    ]
    
    for case_name, sel_bits, dout_vals in cases:
        # Convert to index
        idx = sel_bits[0] * 8 + sel_bits[1] * 4 + sel_bits[2] * 2 + sel_bits[3]
        print(f"  {case_name} -> sel={sel_bits} -> index={idx} -> dout={dout_vals}")
    
    print()
    
    # Test our parsing
    sel_vars, dout_lists = parse_verilog_example()
    
    print("Parsed dout_lists:")
    for i, dout_list in enumerate(dout_lists):
        print(f"  dout[{i}]: {dout_list}")
    
    print()
    
    # Test individual trees
    for i, dout_list in enumerate(dout_lists):
        tree = BinaryDecisionTree(sel_vars, dout_list)
        print(f"dout[{i}] expression: {tree.get_expression()}")
    
    print()

def test_bit_ordering():
    """Test bit ordering and indexing."""
    print("Testing bit ordering")
    print("=" * 20)
    
    # For 4-bit sel[3:0], verify index calculation
    print("Index calculation verification:")
    for sel3 in [0, 1]:
        for sel2 in [0, 1]:
            for sel1 in [0, 1]:
                for sel0 in [0, 1]:
                    idx = sel3 * 8 + sel2 * 4 + sel1 * 2 + sel0
                    binary = f"{sel3}{sel2}{sel1}{sel0}"
                    print(f"  sel[3:0]={binary} -> index={idx}")
    print()

if __name__ == "__main__":
    test_simple_case()
    test_bit_ordering()
    debug_verilog_example()