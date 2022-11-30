import sys

IN_COLAB = 'google.colab' in sys.modules
DIR_PATH = ""

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/dirve")
    DIR_PATH = r"/content/dirve/MyDrive/研究所/Data/dblp/"
    sys.path.append('/content/dirve/MyDrive/Colab Notebooks/package')
else:
    DIR_PATH = r"D:\\論文實驗\\data\\dblp\\"
    sys.path.append('D:\\論文實驗\\package')
    sys.path.append("D:\\論文實驗\\env\\Lib\\site-packages")

import importlib
import networkx as nx 
import unittest
from coupon import Coupon
from social_graph import SN_Graph
from model import DiffusionModel
import logging

if __name__ == "__main__":
    
    testRunner = unittest.TextTestRunner()
    suite = unittest.defaultTestLoader.discover("./test/")
    testRunner.run(suite)
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    
    # graph = SN_Graph()
    # graph.construct(DIR_PATH + "edges", DIR_PATH + "topic_nodes.csv")
    # subgraph = graph.sampling_subgraph(10)
    # nx.write_gml(subgraph, DIR_PATH + "sample10_graph.gml")
    
    
    graph = SN_Graph()
    graph.add_edge(0, 1, weight=0.01, is_tested=False)
    graph.add_edge(0, 2, weight=1, is_tested=False)
    for node in graph:
        graph.nodes[node]['desired_set'] = None
        graph.nodes[node]['adopted_set'] = None
            
    
    topic = {
            '0': [0.82, 0.19],
            '1': [0.63, 0.37],
            '2': [0.5, 0.5]
        }
    price = [60,260,70]
    
    model = DiffusionModel(
        "test",
        graph, 
        {"price": price, "topic": topic},
        [Coupon(180, [0], 20, [0,1]),]
    )

    
    model.diffusion()
    model.save(DIR_PATH + "checkpoint/")