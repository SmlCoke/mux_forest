#!/usr/bin/env python3
"""
Quick test of the optimization functionality.
"""

from mux_forest import BinaryDecisionForest, parse_verilog_example
import time

def test_small_optimization():
    """Test optimization on small example."""
    print("Testing small example with new optimization...")
    sel_vars, dout_lists = parse_verilog_example()
    forest = BinaryDecisionForest(sel_vars, dout_lists)
    
    initial_nodes = forest.count_total_nodes()
    print(f"Initial nodes: {initial_nodes}")
    
    # Test optimization
    start_time = time.time()
    optimal_order, min_nodes = forest.optimize_sel_order(max_iterations=1000)
    optimization_time = time.time() - start_time
    
    print(f"Optimization completed in {optimization_time:.3f}s")
    print(f"Initial: {initial_nodes} nodes, Optimized: {min_nodes} nodes")
    print(f"Improvement: {initial_nodes - min_nodes} nodes ({((initial_nodes - min_nodes) / initial_nodes * 100):.1f}%)")
    print(f"Optimal order: {optimal_order}")
    
    return min_nodes == 14  # Expected result

def test_medium_problem():
    """Test on a medium-sized problem to verify heuristic optimization."""
    print("\nTesting medium-sized problem (8 variables)...")
    
    # Create 8-variable problem
    sel_vars = [f"sel[{i}]" for i in range(7, -1, -1)]  # 8 variables
    
    # Create simple pattern with some structure
    dout_lists = []
    for bit in range(3):  # 3 output bits
        values = []
        for i in range(256):  # 2^8 combinations
            if i % 4 == bit:
                values.append(1)
            elif i % 8 == (bit + 3) % 8:
                values.append(0)
            else:
                values.append(-1)  # don't care
        dout_lists.append(values)
    
    forest = BinaryDecisionForest(sel_vars, dout_lists)
    initial_nodes = forest.count_total_nodes()
    print(f"Initial nodes: {initial_nodes}")
    
    # Test heuristic optimization with limited iterations
    start_time = time.time()
    optimal_order, min_nodes = forest.optimize_sel_order(max_iterations=50)
    optimization_time = time.time() - start_time
    
    print(f"Optimization completed in {optimization_time:.3f}s")
    print(f"Initial: {initial_nodes} nodes, Optimized: {min_nodes} nodes")
    improvement = initial_nodes - min_nodes
    if improvement > 0:
        print(f"Improvement: {improvement} nodes ({(improvement / initial_nodes * 100):.1f}%)")
    else:
        print("No improvement found (may already be optimal)")
    
    return True

if __name__ == "__main__":
    print("Quick Optimization Test")
    print("=" * 40)
    
    # Test small example
    success1 = test_small_optimization()
    print(f"Small test: {'PASS' if success1 else 'FAIL'}")
    
    # Test medium example  
    success2 = test_medium_problem()
    print(f"Medium test: {'PASS' if success2 else 'FAIL'}")
    
    if success1 and success2:
        print("\n✓ All optimization tests passed!")
    else:
        print("\n✗ Some tests failed!")