import unittest
import networkx as nx

from package.itemset import ItemsetFlyweight, Itemset
from package.social_graph import SN_Graph 
from package.coupon import Coupon
from package.user_proxy import UsersProxy

class TestSN_Graph(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSN_Graph, self).__init__(*args, **kwargs)

    def setUp(self) -> None:
        
        G = SN_Graph(located=False)
        nx.add_path(G, [0,1,2,3,6,7])
        nx.add_path(G, [1,4,2])
        nx.add_path(G, [0,2])
        nx.add_path(G, [5,3])
        G.initAttr()

        self._graph = G
        return super().setUp()

    def test_maxExpected(self):
        G = self._graph
        max_expected_graph, max_expected_length = SN_Graph.compile_max_product_graph(G, [0,1])
        self.assertAlmostEqual(max_expected_length[3], 1/6, msg="The maximum expected influence probability is wrong.")
        self.assertFalse((5 in max_expected_length), msg="The maximum expected graph is wrong.")
        self.assertFalse(max_expected_graph.has_edge(4,2), msg="The maximum expected graph is wrong.")
