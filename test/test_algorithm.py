from package.social_graph import SN_Graph
from package.topic import TopicModel
from package.algorithm import Algorithm
from package.itemset import ItemRelation, ItemsetFlyweight
from package.model import DiffusionModel
from package.coupon import Coupon
from package.tag import TagRevenue, TagActiveNode

import unittest
import pandas as pd
import networkx as nx

class TestAlgorithm(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super(TestAlgorithm, self).__init__(*args, **kwargs)
    
    def setUp(self) -> None:
        TOPICS = {
            "Node": {
                0: [0.9, 0.1, 0.0],
                1: [0.2, 0.8, 0.0],
                2: [0.5, 0.2, 0.3],
                3: [0.6, 0.3, 0.1],
                4: [0.7, 0.2, 0.1],
                5: [0.45, 0.65, 0],

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
            
        relation = ItemRelation(RELATION)
        topic = TopicModel(3, TOPICS["Node"], TOPICS["Item"])
        itemset = ItemsetFlyweight(PRICES, topic, relation)
        graph = SN_Graph(topic.getNodesTopic(), located=False)
        nx.add_path(graph, [0,1,2,3])
        nx.add_path(graph, [1,4,2])
        graph.add_edge(0,2)
        graph.add_edge(5,3)
        graph.initAttr()
        
        graph.edges[0,2]["is_active"] = False
        graph.edges[1,2]["is_active"] = True
        graph.edges[2,3]["is_active"] = True
        graph.edges[4,2]["is_active"] = True

        self._model = DiffusionModel("TestGreedy", graph=graph, itemset=itemset, coupons=[], threshold=10**(-6))
        self._model.selectSeeds(2)
        self._model.allocate(self._model.getSeeds(), [itemset["Galaxy"], itemset["iPhone"]])
        return super().setUp()
    
    @unittest.skip("The execution time is too expensive")
    def test_greedy(self):
        itemset = self._model.getItemsetHandler()

        coupons = [Coupon(280, itemset["iPhone Galaxy"], 80, itemset["Galaxy"]),
                   Coupon(280, itemset["iPhone AirPods"], 50, itemset["AirPods"])]
        
        algo = Algorithm(self._model, 2)
        for k in range(2, 5):
            outputCoupon, tagger = algo.simulation(coupons)
            self.assertListEqual([coupons[0]], outputCoupon, "The output of coupons is not correct.")
            self.assertEqual(tagger["TagRevenue"].expected_amount(), 1770, "Revenue is not correct.")
            self.assertEqual(tagger["TagActiveNode"].expected_amount(), 3.5, "The number of expected active node is not correct.")

        '''
            While coupon[0] is in the output, the revenue is reduced if we append coupon[1]
        '''
