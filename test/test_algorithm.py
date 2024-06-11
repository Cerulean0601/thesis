from package.social_graph import SN_Graph
from package.cluster_graph import ClusterGraph
from package.topic import TopicModel
from package.algorithm import Algorithm
from package.itemset import ItemRelation, ItemsetFlyweight
from package.model import DiffusionModel
from package.coupon import Coupon

import unittest
import pandas as pd
import networkx as nx
import math

class TestAlgorithm(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(TestAlgorithm, self).__init__(*args, **kwargs)
    
    def setUp(self) -> None:
        TOPICS = {
            "Cluster": {
                '1': [0.4139, 0.2764, 0.3097],
                '2, 3, 5': [0.31980000000000003, 0.31626666666666664, 0.3639333333333333],
                '6': [0.0213, 0.5398, 0.4389],
                '0, 9': [0.11779999999999999, 0.47209999999999996, 0.4101],
                '1, 3': [0.3639, 0.2671, 0.369],
                '4': [0.4535, 0.5131, 0.0334],
                '8': [0.0816, 0.0634, 0.855],
                '3': [0.3139, 0.2578, 0.4283]

            },
            "Item": {
                "iPhone": [0.7, 0.0, 0.3],
                "AirPods": [0.9, 0.0, 0.1],
                "Galaxy": [0.0, 0.8, 0.2],
            }
        }
        PRICES = {
            "iPhone": 260,
            "AirPods": 60,
            "Galaxy": 500,
        }
        RELATION = pd.DataFrame.from_dict({
                    "iPhone":{
                        "AirPods":10,
                        "Galaxy":-5
                    },
                    "AirPods":{
                        "iPhone":1,
                        "Galaxy":0,
                    },
                    "Galaxy":{
                        "iPhone":-8,
                        "AirPods":1,
                    }
                    })
        
        edges = {('1', '2, 3, 5'): 2.3333333333333335,
            ('1', '6'): 1.0,
            ('2, 3, 5', '0, 9'): 2.0,
            ('2, 3, 5', '1, 3'): 0.8333333333333333,
            ('2, 3, 5', '4'): 0.75,
            ('2, 3, 5', '8'): 0.75,
            ('0, 9', '1'): 0.5,
            ('1, 3', '2, 3, 5'): 2.3333333333333335,
            ('1, 3', '4'): 0.5,
            ('1, 3', '6'): 1.0,
            ('4', '3'): 0.3333333333333333,
            ('3', '4'): 0.5}
        
        relation = ItemRelation(RELATION)
        topic = TopicModel(3, TOPICS["Cluster"], TOPICS["Item"])
        self._itemset = ItemsetFlyweight(PRICES, topic, relation)
        self._graph = ClusterGraph(theta = 0.9, depth = 4, located=False)
        self._graph.add_edges_from([(u, v) for u, v in edges.keys()])
        self._graph.initAttr()
        nx.set_node_attributes(self._graph, TOPICS["Cluster"], "topic")
        nx.set_edge_attributes(self._graph, edges, "weight")
        nx.set_edge_attributes(self._graph, True, "is_active")
        self._model = DiffusionModel(self._graph, self._itemset)
        self._model._seeds = ["1"]
        self._algo = Algorithm(self._model, 1, cluster_theta = 0.9, depth = 4)
        return super().setUp()
    
    def test_locally_estimate(self):
        self._model.allocate(["1"], [self._itemset["Galaxy"]])
        min_discount = math.ceil(self._model._user_proxy._min_discount("1", "Galaxy", "Galaxy iPhone"))
        coupon = Coupon(accThreshold=self._itemset["Galaxy"].price, 
                        accItemset=self._itemset["Galaxy"], 
                        discount=min_discount,
                        disItemset=self._itemset["iPhone"])
        level_clusters = list(self._graph._level_travesal(self._model.getSeeds(), depth=5))
        revenue = self._algo._locally_estimate(level_clusters[0], level_clusters[1], coupon)
        self.assertEqual(revenue, 760.0)

    def test_globally_estimate(self):
        self._model.allocate(["1"], [self._itemset["Galaxy"]])
        min_discount = math.ceil(self._model._user_proxy._min_discount("1", "Galaxy", "Galaxy iPhone"))
        coupon = Coupon(accThreshold=self._itemset["Galaxy"].price, 
                        accItemset=self._itemset["Galaxy"], 
                        discount=min_discount,
                        disItemset=self._itemset["iPhone"])
        self.assertAlmostEqual(self._algo._globally_estimate(coupon), 4937.083333333334)

    # @unittest.skip("The execution time is too expensive")
    def test_greedy(self):
        TOPICS = {
            "Node": {
                0: [0.9, 0.1, 0.0],
                1: [0.2, 0.8, 0.0],
                2: [0.5, 0.2, 0.3],
                3: [0.6, 0.3, 0.1],
                4: [0.7, 0.2, 0.1],
                5: [0.45, 0.65, 0]
            }
        }
        
        itemset = self._model.getItemsetHandler()

        graph = SN_Graph(TOPICS["Node"], located=False)
        nx.add_path(graph, [0,1,2,3])
        nx.add_path(graph, [1,4,2])
        graph.add_edge(0,2)
        graph.add_edge(5,3)
        graph.initAttr()

        graph.edges[0,2]["is_active"] = False
        graph.edges[1,2]["is_active"] = True
        graph.edges[2,3]["is_active"] = True
        graph.edges[4,2]["is_active"] = True

        self._model.setGraph(graph)
        self._model.selectSeeds(2)
        self._model.allocate(self._model.getSeeds(), [itemset["Galaxy"], itemset["iPhone"]])

        coupons = [Coupon(280, itemset["iPhone Galaxy"], 80, itemset["Galaxy"]),
                   Coupon(280, itemset["iPhone AirPods"], 50, itemset["AirPods"])]
        
        algo = Algorithm(self._model, k=2, depth=0)
        for k in range(2, 5):
            outputCoupon, tagger = algo.simulation(coupons)
            self.assertListEqual([coupons[0]], outputCoupon, "The output of coupons is not correct.")
            self.assertEqual(tagger["TagRevenue"].expected_amount(), 1770, "Revenue is not correct.")
            self.assertEqual(tagger["TagActiveNode"].expected_amount(), 3.5, "The number of expected active node is not correct.")

        '''
            While coupon[0] is in the output, the revenue is reduced if we append coupon[1]
        '''
