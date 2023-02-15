import os
import unittest
import networkx as nx

# const.
DATA_ROOT = os.path.join("D:" + os.sep, "論文實驗", "data")
DBLP_PATH = os.path.join(DATA_ROOT, "dblp")
AMAZON_PATH = os.path.join(DATA_ROOT, "amazon")


def test():
    testRunner = unittest.TextTestRunner()
    suite = unittest.defaultTestLoader.discover("./test/")
    testRunner.run(suite)

import sys
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
sys.path.append('D:\\論文實驗\\package')

import pandas as pd

from model import DiffusionModel
from topic import TopicModel
from social_graph import SN_Graph
from itemset import ItemsetFlyweight, ItemRelation
from coupon import Coupon
from utils import getItemsPrice
from algorithm import Algorithm

NUM_TOPICS = 3
TOPICS = {
    "Node": {
        "0": [0.9, 0.1, 0.0],
        "1": [0.2, 0.8, 0.0],
        "2": [0.2, 0.2, 0.6],
        "3": [0.2, 0.4, 0.4],
    },
    "Item": {
        "iPhone": [0.8, 0.0, 0.2],
        "AirPod": [0.7, 0.0, 0.3],
        "Galaxy": [0.0, 0.8, 0.2],
    }
}
PRICES = {
    "iPhone": 260,
    "AirPod": 60,
    "Galaxy": 200,
}
RELATION = pd.DataFrame.from_dict({
            "iPhone":{
                "AirPod":10,
                "Galaxy":-5
            },
            "AirPod":{
                "iPhone":1,
                "Galaxy":0,
            },
            "Galaxy":{
                "iPhone":-8,
                "AirPod":1,
            }
            })

        
if __name__ == '__main__':
        
    test()
    
    topicModel = TopicModel(NUM_TOPICS, TOPICS["Node"], TOPICS["Item"])

    # graph = SN_Graph(TOPICS["Node"])
    # graph.add_edge("0", "1")
    # graph.add_edge("0", "2")

    graph = SN_Graph.transform(nx.diamond_graph(), TOPICS["Node"])

    
    relation = ItemRelation(RELATION)
    itemset = ItemsetFlyweight(PRICES, topicModel.getItemsTopic(), relation)
    items = [itemset[id] for id in list(itemset.PRICE.keys())]

    model = DiffusionModel("dblp_amazon", graph, itemset)
    model._selectSeeds(2)
    model.allocate(model._seeds, items)
    
    algo = Algorithm(model)
    tagger = algo._preporcessing()
    print(tagger._next.table)
    print(tagger._next._next.table)
    # model.diffusion()

