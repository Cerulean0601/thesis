import unittest
import networkx as nx

from package.itemset import ItemsetFlyweight, Itemset
from package.social_graph import SN_Graph 
from package.coupon import Coupon
from package.user_proxy import UsersProxy

class TestSN_Graph(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSN_Graph, self).__init__(*args, **kwargs)

    def test_maxExpected(self):

        G = SN_Graph(located=False)
        nx.add_path(G, [0,1,2,3,6,7])
        nx.add_path(G, [4,2])
        nx.add_path(G, [0,2])
        nx.add_path(G, [5,3])
        nx.add_path(G, [4,1])
        G.initAttr() 

        max_expected_graph, max_expected_length = SN_Graph.compile_max_product_graph(G, [0,1])
        self.assertAlmostEqual(max_expected_length[3], 1/6, msg="The maximum expected influence probability is wrong.")
        self.assertFalse((5 in max_expected_length), msg="The maximum expected graph is wrong.")
        self.assertFalse(max_expected_graph.has_edge(4,2), msg="The maximum expected graph is wrong.")

    def test_max_product_path(self):
        # Test with a graph containing only one node and one seed
        graph = SN_Graph(located=False)
        graph.add_node("A")
        seeds = ["A"]
        graph._initAllEdges()

        max_expected, path = SN_Graph.max_product_path(graph, seeds)
    
        self.assertEqual(max_expected, {"A": 1})
        self.assertEqual(path, {"A": ["A"]})

        # Test with a graph containing multiple node and no seeds
        seeds = []
        with self.assertRaises(ValueError):
            SN_Graph.max_product_path(graph, seeds)

        # Test with a graph containing multiple nodes and one seed
        seeds = ["A"]
        graph.add_edge("A", "B", weight=0.5)
        
        max_expected, path = SN_Graph.max_product_path(graph, seeds)
    
        self.assertEqual(max_expected, {"A": 1, "B": 0.5})
        self.assertEqual(path, {"A": ["A"], "B": ["A","B"]})

        # Test with a graph containing multiple nodes and multiple seeds
        graph.add_edge("A", "B", weight=0.5)
        graph.add_edge("B", "C", weight=0.4)
        graph.add_edge("C", "D", weight=0.3)
        graph.add_edge("D", "E", weight=0.2)
        seeds = ["A", "D"]
    
        max_expected, path = SN_Graph.max_product_path(graph, seeds)
    
        expected_weights = {"A": 1, "B": 0.5, "C": 0.2, "D": 1, "E": 0.2}
        for node, weight in expected_weights.items():
            self.assertAlmostEqual(max_expected[node], weight)
        self.assertEqual(path, {"A": ["A"], "B": ["A","B"], "C": ["A","B","C"], "D": ["D"], "E": ["D","E"]})
        
        # Test with a graph containing multiple nodes and no edges
        graph = SN_Graph(located=False)
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        seeds = ["A"]
        graph._initAllEdges()

        max_expected, path = SN_Graph.max_product_path(graph, seeds)

        self.assertEqual(max_expected, {"A": 1})
        self.assertEqual(path, {"A": ["A"]})

