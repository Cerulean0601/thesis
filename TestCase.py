import os
import unittest
import networkx as nx
import sys
from time import time

# CONSTANT
DATA_ROOT = "./data/"
DBLP_PATH = DATA_ROOT + "dblp/"
AMAZON_PATH = DATA_ROOT + "amazon/"
FACEBOOK_PATH = DATA_ROOT + "facebook/"


def test():
    testRunner = unittest.TextTestRunner()
    suite = unittest.defaultTestLoader.discover("./test/")
    testRunner.run(suite)

import sys
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
sys.path.append('./package/')

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
        "2": [0.8, 0.2, 0.0],
        "3": [0.2, 0.4, 0.4],
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

if __name__ == '__main__':    
    
    test()
    
    nodes = []
    with open(r"./data/facebook/data/edges") as f:
        for line in f:
            src, dst = line.split(",")
            dst = dst[:-1] if dst[-1] == "\n" else dst
            if src not in nodes:
                nodes.append(src)
            if dst not in nodes:
                nodes.append(dst)

    topicModel = TopicModel(NUM_TOPICS)
    topicModel.setItemsTopic(TOPICS["Item"])
    topicModel.randomTopic(nodes_id = nodes)
    
    graph = SN_Graph.construct(r"data/facebook/data/edges", topicModel, located=False)

    relation = ItemRelation(RELATION)
    itemset = ItemsetFlyweight(PRICES, topicModel, relation)

    model = DiffusionModel("facebook", graph, itemset)
    model.selectSeeds(len(PRICES.keys()))

    algo = Algorithm(model)
    
    simluation_times = 10
    start_time = time()
    for i in range(simluation_times):
        candidatedCoupons = algo.genAllCoupons(1)
        coupons = algo.simulation(candidatedCoupons)
    end_time = time()
    print("Runtimes: %.3f", (end_time - start_time)/simluation_times)

    print("candidatedCoupons {0}".format([str(c) for c in candidatedCoupons]))
    print("coupons {0}".format([str(c) for c in coupons]))