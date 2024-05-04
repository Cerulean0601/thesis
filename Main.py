import unittest
from time import time, ctime, sleep
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

#import logging

#logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

import pandas as pd
import networkx as nx
from copy import deepcopy

from package.tag import Tagger, TagRevenue, TagActiveNode
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

def main():
    items = read_items(AMAZON_PATH + "/sample_items.csv")

    topicModel = TopicModel(NUM_TOPICS)
    topicModel.read_topics(node_file=FACEBOOK_PATH + "/nodes_with_" + str(NUM_TOPICS) + "_topic.csv",
                           items_file=AMAZON_PATH + "/items_with_" + str(NUM_TOPICS) + "_topic.csv")

    graph = SN_Graph.construct(FACEBOOK_PATH + "/edges", topicModel, located=False)
    relation = ItemRelation()
    relation.construct(AMAZON_PATH + "/sample_items.csv")
    itemset = ItemsetFlyweight(getItemsPrice(AMAZON_PATH + "/sample_items.csv"), topicModel, relation)

    model = DiffusionModel( graph, itemset, threshold=10**(-5), name="amazon in dblp")
    seed_size = min(itemset.size, graph.number_of_nodes())
    seeds = model.selectSeeds(seed_size)
    model.allocate(seeds, [itemset[asin] for asin in itemset.PRICE.keys()])
    algo = Algorithm(model, 20, depth=0)

    start_time = time()
    
    subgraph = graph.bfs_sampling(algo._max_expected_len, roots=model.getSeeds())
    for s in seeds:
        for attr, value in graph.nodes[s].items():
            subgraph.nodes[s][attr] = value
    print(len(subgraph))
    print(len(subgraph.edges))
    algo.setGraph(subgraph)

    for d in range(0,2):
        
        algo._depth = d
        coupons = algo.genSelfCoupons()
        print([str(c) for c in coupons])
        # end_time = time()

        # print("time:{}".format(end_time-start_time))
        # model.setCoupons(coupons)
        # tagger = Tagger()
        # tagger.setNext(TagRevenue(graph, model.getSeeds(), algo._max_expected_len))
        # tagger.setNext(TagActiveNode())

        # print("Simulate Diffusion...")
        # start = time()
        # simulationTimes = 1
        # algo.setGraph(graph)
        # for i in range(simulationTimes):
        #     g = model.getGraph()
        #     g.initAttr()
        #     model.diffusion(tagger)
        # print(time()-start)
        # performanceFile = r"./result/Self.txt"
        # with open(performanceFile, "a") as record:

        #     tagger["TagRevenue"].avg(simulationTimes)
        #     tagger["TagActiveNode"].avg(simulationTimes)

        #     record.write("{0},runtime={1},revenue={2},expected_revenue={3},active_node={4},expected_active_node={5}\n".format(
        #         ctime(end_time),
        #         (end_time - start_time),
        #         tagger["TagRevenue"].amount(),
        #         tagger["TagRevenue"].expected_amount(),
        #         tagger["TagActiveNode"].amount(),
        #         tagger["TagActiveNode"].expected_amount(),
        #         ))
            
        #     for c in coupons:
        #         record.write(str(c) + "\n")
        #     record.write("\n")
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