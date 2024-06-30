import unittest
from time import time, ctime, sleep
from notify_run import Notify
import numpy as np

# CONSTANT
DATA_ROOT = "./data"
DBLP_PATH = DATA_ROOT + "/dblp"
AMAZON_PATH = DATA_ROOT + "/amazon"
AMAZON_NETWORK_PATH = DATA_ROOT + "/amazon_network"
FACEBOOK_PATH = DATA_ROOT + "/facebook_smallest"
NOTIFY_ENDPOINT = r"https://notify.run/O6EfLmG6Tof1s5DljYB7"
CLUB_PATH = DATA_ROOT + "/Karate Club Network"

def test():
    testRunner = unittest.TextTestRunner()
    suite = unittest.defaultTestLoader.discover("./test/")
    testRunner.run(suite)

#import logging

#logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

import pandas as pd
import networkx as nx

from package.coupon import Coupon
from package.model import DiffusionModel
from package.topic import TopicModel
from package.social_graph import SN_Graph
from package.itemset import ItemsetFlyweight, ItemRelation
from package.utils import getItemsPrice, read_items
from package.algorithm import Algorithm

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

GRAPH = FACEBOOK_PATH
def main():
    items = read_items(AMAZON_PATH + "/sample_items.csv")

    topicModel = TopicModel(NUM_TOPICS)
    topicModel.read_topics(node_file=GRAPH + "/nodes_with_" + str(NUM_TOPICS) + "_topic.csv",
                            items_file=AMAZON_PATH + "/items_with_" + str(NUM_TOPICS) + "_topic.csv")

    graph = SN_Graph.construct(GRAPH + "/edges", topicModel, located=False)
    relation = ItemRelation()
    relation.construct(AMAZON_PATH + "/sample_items.csv", substitute_coff=0.8, complementary_coff=1.2)

    # 選擇可以產生coupon的商品
    # for_coupon_items = []
    # for asin, attr in items.items():
    #     if attr["for_coupon"] == '1':
    #         for_coupon_items.append(asin)
    itemset = ItemsetFlyweight(prices=getItemsPrice(AMAZON_PATH + "/sample_items.csv"), 
                               topic=topicModel, 
                               relation=relation)

    model = DiffusionModel( graph, itemset, threshold=10**(-5), name="amazon in dblp")
    seed_size = min(itemset.size, graph.number_of_nodes())
    seeds = model.selectSeeds(seed_size)

    model.allocate(seeds, [itemset[asin] for asin in itemset.PRICE.keys()])

    simluation_times = 1000
    algo = Algorithm(model, simulationTimes = simluation_times)

    performanceFile = r"./result/test.txt"
    candidatedCoupons = algo.genAllCoupons(5)
    # candidatedCoupons = algo.genDiscountItemCoupons(list(np.arange(0.1, 1, 0.1)))
    # candidatedCoupons = algo.genFullItemCoupons(min(itemset.PRICE.values()), 5, list(np.arange(0.1, 1, 0.1)))
    generator = algo.optimalAlgo(candidatedCoupons, 1)
    
    # coupons = [Coupon(24.95, itemset["B000DIMTEY"], 4.11, itemset["B00005NN16"]),
    #            Coupon(18.99, itemset["B000JX5JGI"], 24.58, itemset["B000DIMTEY"]),
    #            Coupon(5.99, itemset["B00005NN16"], 18.68, itemset["B000JX5JGI"]),
    #            Coupon(5.99, itemset["B00005NN16"], 24.71, itemset["B000DIMTEY"])]

    # generator = [algo._parallel(3, [coupons[0], coupons[1], coupons[2], coupons[3]])]
    # generator = [(coupons,algo._parallel(3, coupons))]
    # generator = [algo._parallel(-1, [])]
    start_time = time()
    max_revenue = 0
    k = 0
    for outputCoupons,tagger in generator:
        end_time = time()
                
        with open(performanceFile, "a") as record:

            record.write("{0},runtime={1},revenue={2},expected_revenue={3},active_node={4},expected_active_node={5},k={6}\n".format(
                ctime(end_time),
                (end_time - start_time),
                tagger["TagRevenue"].amount(),
                tagger["TagRevenue"].expected_amount(),
                tagger["TagActiveNode"].amount(),
                tagger["TagActiveNode"].expected_amount(),
                k,
                ))

            for c in outputCoupons:
                record.write(str(c) + "\n")
            record.write("\n")
        k += 1        

if __name__ == '__main__':

    # test()
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