import unittest
import networkx as nx

from package.topic import TopicModel
from package.itemset import ItemsetFlyweight, Itemset
from package.cluster_graph import ClusterGraph 
from package.social_graph import SN_Graph
from package.coupon import Coupon
from package.user_proxy import UsersProxy

class TestClusterGraph(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestClusterGraph, self).__init__(*args, **kwargs)

    def test_cluster(self):
        TOPICS = {
            "Item": {
                "iPhone": [0.7, 0.0, 0.3],
                "AirPods": [0.9, 0.0, 0.1],
                "Galaxy": [0.0, 0.8, 0.2],
            },
            "Node": {
                0:[0.0957, 0.4980, 0.4063],
                1:[0.4139, 0.2764, 0.3097],
                4:[0.4535, 0.5131, 0.0334],
                5:[0.2464, 0.3600, 0.3936],
                9:[0.1399, 0.4462, 0.4139],
                2:[0.3991, 0.3310, 0.2699],
                6:[0.0213, 0.5398, 0.4389],
                7:[0.7189, 0.1738, 0.1073],
                8:[0.0816, 0.0634, 0.8550],
                3:[0.3139, 0.2578, 0.4283]
            }
        }
        
        G = nx.DiGraph()
        G.add_edges_from([(0,1),
                        (1,2),(1,3),(1,5),(1,6),
                        (2,0),(2,1),(2,3),(2,4),(2,8),(2,9),
                        (3,4),
                        (4,3),
                        (5,8)])
        
        topic = TopicModel(3, itemsTopic=TOPICS["Item"], nodesTopic=TOPICS["Node"])
        H = SN_Graph.transform(G, topic.getNodesTopic())
        cluster_graph = ClusterGraph(H, [1], 0.9, 6)
        topics = {1: [0.4139, 0.2764, 0.3097],
                    '2, 3, 5': [0.31980000000000003, 0.31626666666666664, 0.3639333333333333],
                    '6': [0.0213, 0.5398, 0.4389],
                    '0, 9': [0.11779999999999999, 0.47209999999999996, 0.4101],
                    '1, 3': [0.3639, 0.2671, 0.369],
                    '4': [0.4535, 0.5131, 0.0334],
                    '8': [0.0816, 0.0634, 0.855],
                    '1': [0.4139, 0.2764, 0.3097],
                    '3': [0.3139, 0.2578, 0.4283]}

        self.assertDictEqual(nx.get_node_attributes(cluster_graph, "topic"), topics)

        weights = {(1, '2, 3, 5'): 2.3333333333333335,
                    (1, '6'): 1.0,
                    ('2, 3, 5', '0, 9'): 2.0,
                    ('2, 3, 5', '1, 3'): 0.8333333333333333,
                    ('2, 3, 5', '4'): 0.75,
                    ('2, 3, 5', '8'): 0.75,
                    ('0, 9', '1'): 0.5,
                    ('1, 3', '2, 3, 5'): 2.3333333333333335,
                    ('1, 3', '4'): 0.5,
                    ('1, 3', '6'): 1.0,
                    ('4', '3'): 0.3333333333333333,
                    ('1', '2, 3, 5'): 2.3333333333333335,
                    ('1', '6'): 1.0,
                    ('3', '4'): 0.5}
        
        self.assertDictEqual(weights, nx.get_edge_attributes(cluster_graph, "weights"))