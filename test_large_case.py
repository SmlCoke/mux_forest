#!/usr/bin/env python3
"""
Test script for large-scale pmux optimization using the tc.txt file.
"""

import re
import time
from mux_forest import BinaryDecisionForest


def parse_verilog_file(filename):
    """
    Parse a Verilog file with pmux cases and extract sel_vars and dout_lists.
    
    Args:
        filename: Path to the Verilog file
        
    Returns:
        Tuple of (sel_vars, dout_lists, num_bits, num_cases)
    """
    with open(filename, 'r') as f:
        content = f.read()
    
    # Extract input/output bit widths
    sel_match = re.search(r'input\s+\[(\d+):0\]\s+sel', content)
    dout_match = re.search(r'output\s+\[(\d+):0\]\s+dout', content)
    
    if not sel_match or not dout_match:
        raise ValueError("Could not parse input/output declarations")
    
    sel_width = int(sel_match.group(1)) + 1  # [11:0] means 12 bits
    dout_width = int(dout_match.group(1)) + 1  # [9:0] means 10 bits
    
    print(f"Parsed Verilog: {sel_width}-bit sel, {dout_width}-bit dout")
    
    # Create sel_vars list
    sel_vars = [f"sel[{i}]" for i in range(sel_width-1, -1, -1)]  # MSB first
    
    # Initialize dout_lists with don't care values
    dout_lists = [[-1] * (2**sel_width) for _ in range(dout_width)]
    
    # Extract case statements
    case_pattern = r"(\d+)'b([01]+)\s*:\s*dout\s*=\s*\{([^}]+)\};"
    cases = re.findall(case_pattern, content)
    
    num_cases = 0
    for width_str, binary_str, values_str in cases:
        width = int(width_str)
        if len(binary_str) != width or width != sel_width:
            print(f"Warning: skipping case with width mismatch: {width}'b{binary_str}")
            continue
        
        # Convert binary string to index
        case_index = int(binary_str, 2)
        
        # Parse output values
        values = values_str.strip().split(',')
        if len(values) != dout_width:
            print(f"Warning: skipping case with value count mismatch: {values}")
            continue
        
        # Parse each output value
        for bit_idx, value_str in enumerate(values):
            value_str = value_str.strip()
            
            if value_str == "1'b1":
                dout_lists[dout_width-1-bit_idx][case_index] = 1
            elif value_str == "1'b0":
                dout_lists[dout_width-1-bit_idx][case_index] = 0
            elif value_str == "1'bx":
                dout_lists[dout_width-1-bit_idx][case_index] = -1
            elif value_str.startswith("din[") and value_str.endswith("]"):
                # din[i] -> 2*i+2
                din_idx = int(value_str[4:-1])
                dout_lists[dout_width-1-bit_idx][case_index] = 2*din_idx + 2
            elif value_str.startswith("~din[") and value_str.endswith("]"):
                # ~din[i] -> 2*i+3
                din_idx = int(value_str[5:-1])
                dout_lists[dout_width-1-bit_idx][case_index] = 2*din_idx + 3
            else:
                print(f"Warning: unknown value format: {value_str}")
                dout_lists[dout_width-1-bit_idx][case_index] = -1
        
        num_cases += 1
    
    print(f"Parsed {num_cases} case statements")
    return sel_vars, dout_lists, sel_width, dout_width


def test_large_case_optimization():
    """Test optimization on the large test case."""
    print("=" * 60)
    print("Large-Scale Pmux Optimization Test")
    print("=" * 60)
    
    # Parse the test case file
    try:
        sel_vars, dout_lists, sel_width, dout_width = parse_verilog_file("tc.txt")
    except FileNotFoundError:
        print("Error: tc.txt file not found. Please ensure it's in the current directory.")
        return
    except Exception as e:
        print(f"Error parsing tc.txt: {e}")
        return
    
    print(f"\nProblem size:")
    print(f"  Selection variables: {sel_width} bits")
    print(f"  Output bits: {dout_width} bits") 
    print(f"  Total combinations: {2**sel_width}")
    print(f"  Exhaustive search would require: {sel_width}! = {factorial_approx(sel_width)} permutations")
    
    # Create initial forest
    print(f"\nCreating initial Binary Decision Forest...")
    start_time = time.time()
    forest = BinaryDecisionForest(sel_vars, dout_lists)
    construction_time = time.time() - start_time
    
    initial_nodes = forest.count_total_nodes()
    print(f"Initial construction completed in {construction_time:.2f}s")
    print(f"Initial AIG node count: {initial_nodes}")
    
    # Test optimization with different iteration limits
    iteration_limits = [100, 500, 1000]
    
    for max_iter in iteration_limits:
        print(f"\n" + "-" * 40)
        print(f"Testing optimization with max_iterations={max_iter}")
        print(f"-" * 40)
        
        start_time = time.time()
        optimal_order, min_nodes = forest.optimize_sel_order(max_iterations=max_iter)
        optimization_time = time.time() - start_time
        
        improvement = initial_nodes - min_nodes
        improvement_pct = (improvement / initial_nodes) * 100 if initial_nodes > 0 else 0
        
        print(f"\nOptimization Results:")
        print(f"  Time taken: {optimization_time:.2f}s")
        print(f"  Initial nodes: {initial_nodes}")
        print(f"  Optimized nodes: {min_nodes}")
        print(f"  Improvement: {improvement} nodes ({improvement_pct:.1f}%)")
        print(f"  Optimal order: {optimal_order}")
        
        # Test the optimized forest
        if optimal_order != sel_vars:
            print(f"\nValidating optimized forest...")
            try:
                reordered_dout_lists = forest._reorder_dout_lists(optimal_order)
                optimized_forest = BinaryDecisionForest(optimal_order, reordered_dout_lists)
                optimized_nodes = optimized_forest.count_total_nodes()
                
                if optimized_nodes == min_nodes:
                    print(f"  ✓ Validation successful: {optimized_nodes} nodes")
                else:
                    print(f"  ✗ Validation failed: expected {min_nodes}, got {optimized_nodes}")
            except Exception as e:
                print(f"  ✗ Validation error: {e}")


def factorial_approx(n):
    """Approximate factorial for large numbers."""
    if n <= 20:
        result = 1
        for i in range(1, n + 1):
            result *= i
        return f"{result:,}"
    else:
        # Use Stirling's approximation for display
        import math
        log_fact = n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n)
        return f"~10^{log_fact / math.log(10):.0f}"


def demo_small_example():
    """Demonstrate optimization on the original small example."""
    print("=" * 60)
    print("Small Example Demonstration")
    print("=" * 60)
    
    from mux_forest import parse_verilog_example
    
    sel_vars, dout_lists = parse_verilog_example()
    forest = BinaryDecisionForest(sel_vars, dout_lists)
    
    print(f"Small example: {len(sel_vars)} variables, {len(dout_lists)} output bits")
    print(f"Initial nodes: {forest.count_total_nodes()}")
    
    # This should use exhaustive search
    start_time = time.time()
    optimal_order, min_nodes = forest.optimize_sel_order()
    optimization_time = time.time() - start_time
    
    print(f"Optimization time: {optimization_time:.3f}s")
    print(f"Optimal nodes: {min_nodes}")
    print(f"Optimal order: {optimal_order}")


if __name__ == "__main__":
    # Run small example first
    demo_small_example()
    
    print("\n" + "=" * 80 + "\n")
    
    # Run large case test
    test_large_case_optimization()