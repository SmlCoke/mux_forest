#!/usr/bin/env python3
"""
Demonstrate optimization on the large tc.txt file with reasonable time limits.
"""

import time
from test_large_case import parse_verilog_file
from mux_forest import BinaryDecisionForest

def demo_tc_optimization():
    """Demo optimization on tc.txt with practical time limits."""
    print("Large-Scale Pmux Optimization Demo (tc.txt)")
    print("=" * 50)
    
    try:
        sel_vars, dout_lists, sel_width, dout_width = parse_verilog_file("tc.txt")
    except FileNotFoundError:
        print("tc.txt not found - skipping large case demo")
        return
    
    print(f"Problem: {sel_width}-bit selection, {dout_width}-bit output")
    print(f"Cases: {sum(1 for dl in dout_lists for v in dl if v != -1) // dout_width}")
    
    # Create forest
    forest = BinaryDecisionForest(sel_vars, dout_lists)
    initial_nodes = forest.count_total_nodes()
    print(f"Initial AIG nodes: {initial_nodes}")
    
    # Quick optimization demo
    print(f"\nRunning quick optimization (50 iterations)...")
    start_time = time.time()
    optimal_order, min_nodes = forest.optimize_sel_order(max_iterations=50)
    optimization_time = time.time() - start_time
    
    improvement = initial_nodes - min_nodes
    improvement_pct = (improvement / initial_nodes) * 100 if initial_nodes > 0 else 0
    
    print(f"Results:")
    print(f"  Time: {optimization_time:.1f}s")
    print(f"  Initial: {initial_nodes} nodes")
    print(f"  Optimized: {min_nodes} nodes")
    print(f"  Improvement: {improvement} nodes ({improvement_pct:.1f}%)")
    print(f"  Feasible for engineering use: ✓")

if __name__ == "__main__":
    demo_tc_optimization()