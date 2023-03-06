import os
import unittest
import networkx as nx
import sys

# CONSTANT
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
    
    TOPICS = {
        "Node": {
            "0": [1.0, 0.0, 0.0],
            "1": [0.5, 0.5, 0.0],
        },
        "Item": {
            "iPhone": [0.7, 0.0, 0.3],
            "AirPods": [0.9, 0.0, 0.1],
            "Case": [0.9, 0.0, 0.1],
        }
    }

    PRICES = {
        "iPhone": 260,
        "AirPods": 90,
        "Case": 30,
    }

    RELATION = pd.DataFrame.from_dict({
            "iPhone":{
                "AirPods":10,
                "Case":30
            },
            "AirPods":{
                "iPhone":1,
                "Case":6,
            },
            "Case":{
                "iPhone":3,
                "AirPods":1,
            }
    })

    test()

    topicModel = TopicModel(NUM_TOPICS, TOPICS["Node"], TOPICS["Item"])

    graph = SN_Graph(TOPICS["Node"])
    graph.add_edge("0", "1")

    #graph.add_node("1")
    graph.initAttr()

    # randomGraph = nx.karate_club_graph()
    
    # nodeTopic = dict()
    # for node in randomGraph:
    #     nodeTopic[node] = topicModel.randomTopic()

    # graph = SN_Graph.transform(randomGraph, nodeTopic)
    
    # nx.draw(graph, labels=nx.get_node_attributes(graph, "desired_set"))
    
    relation = ItemRelation(RELATION)
    itemset = ItemsetFlyweight(PRICES, topicModel.getItemsTopic(), relation)
    COUPONS = [Coupon(330, itemset["iPhone AirPods"], 50, itemset["Case"])]
    # items = [itemset[id] for id in list(itemset.PRICE.keys())]

    for ids, obj in itemset:
        print("{0}: {1}".format(ids, obj.topic))

    model = DiffusionModel("dblp_amazon", graph, itemset, COUPONS)
    
    model._seeds = ["1"]
    model.allocate(model._seeds, [itemset["iPhone"]])
    model._nodes["1"] = itemset["iPhone"]
    # algo = Algorithm(model)

    # coupons =  algo.genAllCoupons(50)
    # for c in coupons:
    #     print(c)
    # for c in algo.simulation(coupons):
    #     print(str(c))
    # tagger = algo._preporcessing()
    # print(tagger._next.table)
    # print(tagger._next._next.table)

    model.diffusion()

