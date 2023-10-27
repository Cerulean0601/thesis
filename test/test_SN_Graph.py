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

class TestMaxProductPath(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = SN_Graph(located=False)
        return super().setUp()
    
    def tearDown(self) -> None:
        self.graph.remove_nodes_from(list(self.graph.nodes))
        self.graph.remove_edges_from(list(self.graph.edges))
        return super().tearDown()
    
    def test_multiple_node_no_seed(self):
        seeds = []
        with self.assertRaises(ValueError):
            SN_Graph.max_product_path(self.graph, seeds)
    
    def test_multiple_node_one_seed(self):
        seeds = [0]
        self.graph.add_edge(0, 1, weight=0.5)
        
        max_expected, path = SN_Graph.max_product_path(self.graph, seeds)
    
        self.assertEqual(max_expected, {0: 1, 1: 0.5})
        self.assertEqual(path, {0: [0], 1: [0, 1]})
    
    def test_multiple_node_multiple_seeds(self):
        self.graph.add_edge(0, 1, weight=0.5)
        self.graph.add_edge(1, 2, weight=0.4)
        self.graph.add_edge(2, 3, weight=0.3)
        self.graph.add_edge(3, 4, weight=0.2)
        self.graph.add_edge(0, 4, weight=0.1)
        seeds = [0, 3]
    
        max_expected, path = SN_Graph.max_product_path(self.graph, seeds)
    
        expected_weights = {0: 1, 1: 0.5, 2: 0.2, 3: 1, 4: 0.2}
        for node, weight in expected_weights.items():
            self.assertAlmostEqual(max_expected[node], weight)
        self.assertEqual(path, {0: [0], 1: [0,1], 2: [0,1,2], 3: [3], 4: [3,4]})
    
    def test_multiple_nodes_no_edge(self):
        self.graph.add_node(0)
        self.graph.add_node(1)
        self.graph.add_node(2)
        seeds = [0]

        max_expected, path = SN_Graph.max_product_path(self.graph, seeds)

        self.assertEqual(max_expected, {0: 1})
        self.assertEqual(path, {0: [0]})
    
    def test_multiple_seeds_in_single_componet(self):
        self.graph.add_edge(0, 1, weight=0.2)
        self.graph.add_edge(1, 2, weight=0.3)
        self.graph.add_edge(1, 3, weight=0.4)

        self.graph.add_edge(4, 5, weight=0.7)
        seeds = [0,1]

        max_expected, path = SN_Graph.max_product_path(self.graph, seeds)
        expected_weights = {0:1, 1: 1, 2: 0.3, 3: 0.4}
        for node, weight in expected_weights.items():
            self.assertAlmostEqual(max_expected[node], weight)
            
        self.assertEqual(path, {0:[0], 1: [1], 2: [1,2], 3: [1,3]})
    
    def test_multiple_seeds_in_difference_componets(self):
        self.graph.add_edge(0, 1, weight=0.2)
        self.graph.add_edge(1, 2, weight=0.3)
        self.graph.add_edge(1, 3, weight=0.4)

        self.graph.add_edge(4, 5, weight=0.7)
        seeds = [1,4]

        max_expected, path = SN_Graph.max_product_path(self.graph, seeds)
        expected_weights = {1: 1, 2: 0.3, 3: 0.4, 4: 1, 5: 0.7}
        for node, weight in expected_weights.items():
            self.assertAlmostEqual(max_expected[node], weight)
        self.assertEqual(path, {1: [1], 2: [1,2], 3: [1,3], 4:[4], 5:[4,5]})

