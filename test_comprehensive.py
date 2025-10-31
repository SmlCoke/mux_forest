#!/usr/bin/env python3
"""
Comprehensive test suite for Binary Decision Forest implementation.
"""

import unittest
from mux_forest import BinaryDecisionTree, BinaryDecisionForest, TreeNode, parse_verilog_example

class TestBinaryDecisionTree(unittest.TestCase):
    """Test cases for BinaryDecisionTree class."""
    
    def test_simple_tree_construction(self):
        """Test basic tree construction with 2 variables."""
        sel_vars = ["sel[1]", "sel[0]"]
        dout_values = [0, 1, 2, 3]  # 0=output0, 1=output1, 2=din[0], 3=~din[0]
        
        tree = BinaryDecisionTree(sel_vars, dout_values)
        expr = tree.get_expression()
        
        # Should represent the truth table correctly
        self.assertIn("sel[1]", expr)
        self.assertIn("sel[0]", expr)
        self.assertIn("1'b0", expr)
        self.assertIn("1'b1", expr)
        self.assertIn("din[0]", expr)
        self.assertIn("~din[0]", expr)
    
    def test_all_same_values(self):
        """Test simplification when all outputs are the same."""
        sel_vars = ["sel[1]", "sel[0]"]
        dout_values = [1, 1, 1, 1]  # All outputs are 1
        
        tree = BinaryDecisionTree(sel_vars, dout_values)
        expr = tree.get_expression()
        
        # Should simplify to just 1'b1
        self.assertEqual(expr, "1'b1")
    
    def test_dont_care_simplification(self):
        """Test simplification with don't care values."""
        sel_vars = ["sel[1]", "sel[0]"]
        dout_values = [-1, 1, -1, 1]  # Don't care mixed with 1
        
        tree = BinaryDecisionTree(sel_vars, dout_values)
        expr = tree.get_expression()
        
        # Should simplify significantly
        self.assertIn("1'b1", expr)
    
    def test_terminal_encodings(self):
        """Test all terminal value encodings."""
        tree = BinaryDecisionTree(["sel[0]"], [0, 1])
        
        # Test value to string conversion
        self.assertEqual(tree._value_to_string(0), "1'b0")
        self.assertEqual(tree._value_to_string(1), "1'b1")
        self.assertEqual(tree._value_to_string(-1), "1'bx")
        self.assertEqual(tree._value_to_string(2), "din[0]")
        self.assertEqual(tree._value_to_string(3), "~din[0]")
        self.assertEqual(tree._value_to_string(4), "din[1]")
        self.assertEqual(tree._value_to_string(5), "~din[1]")

class TestBinaryDecisionForest(unittest.TestCase):
    """Test cases for BinaryDecisionForest class."""
    
    def test_forest_construction(self):
        """Test basic forest construction."""
        sel_vars = ["sel[1]", "sel[0]"]
        dout_lists = [
            [0, 1, 0, 1],  # dout[0]
            [1, 0, 1, 0],  # dout[1]
        ]
        
        forest = BinaryDecisionForest(sel_vars, dout_lists)
        expressions = forest.get_expressions()
        
        self.assertEqual(len(expressions), 2)
        for expr in expressions:
            self.assertIn("sel[", expr)
    
    def test_verilog_assigns(self):
        """Test Verilog assign statement generation."""
        sel_vars = ["sel[0]"]
        dout_lists = [[0, 1]]
        
        forest = BinaryDecisionForest(sel_vars, dout_lists)
        assigns = forest.generate_verilog_assigns("dout")
        
        self.assertEqual(len(assigns), 1)
        self.assertIn("assign dout[0] =", assigns[0])
        self.assertTrue(assigns[0].endswith(";"))
    
    def test_node_counting(self):
        """Test AIG node counting."""
        sel_vars = ["sel[0]"]
        dout_lists = [[0, 1], [1, 0]]  # Two complementary outputs
        
        forest = BinaryDecisionForest(sel_vars, dout_lists)
        node_count = forest.count_total_nodes()
        
        # Should have some reasonable number of nodes
        self.assertGreater(node_count, 0)
        self.assertLess(node_count, 20)  # Sanity check
    
    def test_reordering_logic(self):
        """Test sel variable reordering logic."""
        sel_vars = ["sel[1]", "sel[0]"]
        dout_lists = [[0, 1, 2, 3]]
        
        forest = BinaryDecisionForest(sel_vars, dout_lists)
        
        # Test reordering
        new_order = ["sel[0]", "sel[1]"]
        reordered = forest._reorder_dout_lists(new_order)
        
        # Should have same length
        self.assertEqual(len(reordered), 1)
        self.assertEqual(len(reordered[0]), 4)
        
        # Original: index 0 (sel[1]=0,sel[0]=0) -> value 0
        #           index 1 (sel[1]=0,sel[0]=1) -> value 1  
        #           index 2 (sel[1]=1,sel[0]=0) -> value 2
        #           index 3 (sel[1]=1,sel[0]=1) -> value 3
        # Reordered to ["sel[0]", "sel[1]"]:
        #           index 0 (sel[0]=0,sel[1]=0) -> same as old index 0 -> value 0
        #           index 1 (sel[0]=1,sel[1]=0) -> same as old index 1 -> value 1
        #           index 2 (sel[0]=0,sel[1]=1) -> same as old index 2 -> value 2  
        #           index 3 (sel[0]=1,sel[1]=1) -> same as old index 3 -> value 3
        expected = [0, 2, 1, 3]  # Swapped middle two
        self.assertEqual(reordered[0], expected)

class TestVerilogExample(unittest.TestCase):
    """Test the Verilog example from the problem statement."""
    
    def test_example_parsing(self):
        """Test parsing of the example."""
        sel_vars, dout_lists = parse_verilog_example()
        
        self.assertEqual(len(sel_vars), 4)
        self.assertEqual(len(dout_lists), 4)
        
        for dout_list in dout_lists:
            self.assertEqual(len(dout_list), 16)  # 2^4
    
    def test_example_forest(self):
        """Test forest creation from example."""
        sel_vars, dout_lists = parse_verilog_example()
        forest = BinaryDecisionForest(sel_vars, dout_lists)
        
        expressions = forest.get_expressions()
        self.assertEqual(len(expressions), 4)
        
        # All expressions should be valid
        for expr in expressions:
            self.assertIsInstance(expr, str)
            self.assertGreater(len(expr), 0)
    
    def test_example_optimization(self):
        """Test optimization on the example."""
        sel_vars, dout_lists = parse_verilog_example()
        forest = BinaryDecisionForest(sel_vars, dout_lists)
        
        original_nodes = forest.count_total_nodes()
        self.assertGreater(original_nodes, 0)
        
        # Optimization might not improve this specific case,
        # but should at least complete without error
        try:
            optimal_order, min_nodes = forest.optimize_sel_order()
            self.assertEqual(len(optimal_order), 4)
            self.assertLessEqual(min_nodes, original_nodes)
        except Exception as e:
            self.fail(f"Optimization failed: {e}")

class TestTreeNodeEquality(unittest.TestCase):
    """Test TreeNode equality and hashing for reuse detection."""
    
    def test_terminal_node_equality(self):
        """Test equality of terminal nodes."""
        node1 = TreeNode(value=1)
        node2 = TreeNode(value=1)
        node3 = TreeNode(value=0)
        
        self.assertEqual(node1, node2)
        self.assertNotEqual(node1, node3)
        self.assertEqual(hash(node1), hash(node2))
    
    def test_internal_node_equality(self):
        """Test equality of internal nodes."""
        left = TreeNode(value=1)
        right = TreeNode(value=0)
        
        node1 = TreeNode(sel_var="sel[0]", left=left, right=right)
        node2 = TreeNode(sel_var="sel[0]", left=left, right=right)
        node3 = TreeNode(sel_var="sel[1]", left=left, right=right)
        
        self.assertEqual(node1, node2)
        self.assertNotEqual(node1, node3)

if __name__ == '__main__':
    unittest.main(verbosity=2)