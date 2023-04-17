import os
import unittest
import networkx as nx
import sys
from time import time, ctime
from notify_run import Notify

# CONSTANT
DATA_ROOT = "./data"
DBLP_PATH = DATA_ROOT + "/dblp"
AMAZON_PATH = DATA_ROOT + "/amazon"
FACEBOOK_PATH = DATA_ROOT + "/facebook"
NOTIFY_ENDPOINT = r"https://notify.run/O6EfLmG6Tof1s5DljYB7"

def test():
    testRunner = unittest.TextTestRunner()
    suite = unittest.defaultTestLoader.discover("./test/")
    testRunner.run(suite)

import sys
import logging

#logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
sys.path.append('./package/')

import pandas as pd

from model import DiffusionModel
from topic import TopicModel
from social_graph import SN_Graph
from itemset import ItemsetFlyweight, ItemRelation
from coupon import Coupon
from utils import getItemsPrice, read_items
from algorithm import Algorithm

NUM_TOPICS = 5
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
    "iPhone": 50,
    "AirPods": 5,
    "Galaxy": 60,
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

def main():
    items = read_items(AMAZON_PATH + "/sample_items.csv")

    topicModel = TopicModel(NUM_TOPICS)
    topicModel.read_topics(node_file=FACEBOOK_PATH + "/nodes_with_" + str(NUM_TOPICS) + "_topic")
    topicModel.randomTopic(items_id=items.keys())

    graph = SN_Graph.construct(FACEBOOK_PATH + "/edges", topicModel, located=False)

    relation = ItemRelation()
    relation.construct(AMAZON_PATH + "/sample_items.csv")
    itemset = ItemsetFlyweight(getItemsPrice(AMAZON_PATH + "/sample_items.csv"), topicModel, relation)

    model = DiffusionModel("amazon in dblp", graph, itemset, threshold=10**(-5))


    algo = Algorithm(model,0)
    simluation_times = 5
    recordFilename = r"./result/myself.txt"

    for k in range(10):
                
        algo.setLimitCoupon(k)
        for i in range(simluation_times):
            start_time = time()
            candidatedCoupons = algo.genSelfCoupons()
            if k == 0:
                revenue = algo.simulation([])
            else:
                revenue = algo.simulation(candidatedCoupons)
            end_time = time()
        
            with open(recordFilename, "a") as record:
                
                record.write("{0},runtime={1},revenue={2},k={3},times={4}\n".format(
                    ctime(end_time),
                    (end_time - start_time),
                    revenue,
                    k,
                    i
                    ))
                
if __name__ == '__main__':    
    
    test()
    NOTIFY = False

    if NOTIFY:
        notify = Notify(endpoint=NOTIFY_ENDPOINT)
        try:
            main()    
        except Exception as e:
            notify.send("Error: {0}".format(str(e)))
    
    
        notify.send("Done")
    else:
        main()