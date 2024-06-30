import unittest
from time import time, ctime, sleep
from notify_run import Notify
from copy import deepcopy
import numpy as np
from multiprocessing.pool import Pool
from os import cpu_count

# CONSTANT
DATA_ROOT = "./data"
DBLP_PATH = DATA_ROOT + "/dblp"
AMAZON_PATH = DATA_ROOT + "/amazon"
AMAZON_NETWORK_PATH = DATA_ROOT + "/amazon_network"

FACEBOOK_PATH = DATA_ROOT + "/facebook_smallest"
CLUB_PATH = DATA_ROOT + "/Karate Club Network"
NOTIFY_ENDPOINT = r"https://notify.run/O6EfLmG6Tof1s5DljYB7"

def test():
    testRunner = unittest.TextTestRunner()
    suite = unittest.defaultTestLoader.discover("./test/")
    testRunner.run(suite)

#import logging

#logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

import pandas as pd
import networkx as nx
from copy import deepcopy, copy
import numpy as np

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

GRAPH = AMAZON_NETWORK_PATH
NUM_SUBGRAPH = 200
SUB_COFF = 1
COM_COFF = 1
SIM_TIMES = 1000
def main():
    items = read_items(AMAZON_PATH + "/sample_items.csv")

    topicModel = TopicModel(NUM_TOPICS)
    topicModel.read_topics(node_file=GRAPH + "/nodes_with_" + str(NUM_TOPICS) + "_topic.csv",
                           items_file=AMAZON_PATH + "/items_with_" + str(NUM_TOPICS) + "_topic.csv")

    graph = SN_Graph.construct(GRAPH + "/edges", topicModel, located=False)
    relation = ItemRelation()
    relation.construct(AMAZON_PATH + "/sample_items.csv", substitute_coff=SUB_COFF, complementary_coff=COM_COFF)
    itemset = ItemsetFlyweight(getItemsPrice(AMAZON_PATH + "/sample_items.csv"), topicModel, relation)

    model = DiffusionModel(graph, itemset, threshold=10**(-7), name="amazon in dblp")
    seed_size = min(itemset.size, graph.number_of_nodes())
    seeds = model.selectSeeds(seed_size)
    model.allocate(seeds, [itemset[asin] for asin in itemset.PRICE.keys()])
    algo = Algorithm(model)
    
    # NUM_USAGE_CPU = (cpu_count()*2)//3
    # pool = Pool(NUM_USAGE_CPU)
    jaccard_theta = 1
    for subgraph_theta in [0.2,0.4,0.6,0.8,1]:
        for jaccard_theta in [0.2,0.4,0.6,0.8,1]:
            for depth in range(1,2):
                # theta = np.arange(0, 1.1, 0.1)
                theta = [0.2,0.4,0.6,0.8]
                for cluster_theta in theta:

                    start_time = time()
                    coupons = algo.genSelfCoupons(num_sampledGraph=NUM_SUBGRAPH, 
                                        depth=depth, cluster_theta=cluster_theta, 
                                        jaccard_theta=jaccard_theta, 
                                        subgraph_theta=subgraph_theta)
                    end_time = time()

                    print("time:{}".format(end_time-start_time))

                    print("Simulate Diffusion...")
                    start = time()
                    algo.setGraph(graph)
                    
                    tagger = Tagger()
                    tagger.setNext(TagRevenue())
                    tagger.setNext(TagActiveNode())
                    
                    # for coupons in coupons_each_subgraph:
                    model.setCoupons(coupons)

                    seeds_attr = {str(seed): copy(graph.nodes[seed]["desired_set"]) for seed in model.getSeeds()}
                    revenue = 0
                    expected_revenue = 0
                    num_active_nodes = 0
                    expcted_active_nodes = 0

                    graph.resetGraph(seeds_attr)
                    for i in range(SIM_TIMES):
                        tagger = Tagger()
                        tagger.setNext(TagRevenue())
                        tagger.setNext(TagActiveNode())
                        # for n in graph.nodes():
                        #     if graph.nodes[n]["adopted_set"]:
                        #         raise ValueError("")
                        # for u,v in graph.edges():
                        #     if graph.edges[u,v]["is_tested"]:
                        #         raise ValueError("")
                        
                        #     if "is_active" in graph.edges[u,v]:
                        #         if graph.edges[u,v]["is_active"]:
                        #             raise ValueError("")
                            
                        model.diffusion(tagger)
                        graph = model.getGraph()
                        
                        graph.resetGraph(seeds_attr)
                        revenue += tagger["TagRevenue"].amount()
                        expected_revenue += tagger["TagRevenue"].expected_amount()
                        num_active_nodes += tagger["TagActiveNode"].amount()
                        expcted_active_nodes += tagger["TagActiveNode"].expected_amount()
                        

                    revenue /= SIM_TIMES
                    expected_revenue /= SIM_TIMES
                    num_active_nodes /= SIM_TIMES
                    expcted_active_nodes /= SIM_TIMES

                    performanceFile = r"./result/depth_" + str(depth) + ".txt"
                    with open(performanceFile, "a") as record:
                        record.write("{},runtime={},revenue={},expected_revenue={},active_node={},expected_active_node={},cluster_theta={},jaccard_theta={},subgraph_theta={}\n".format(
                            ctime(end_time),
                            (end_time - start_time),
                            revenue,
                            expected_revenue,
                            num_active_nodes,
                            expcted_active_nodes,
                            cluster_theta,
                            jaccard_theta,
                            subgraph_theta
                            ))

                        for c in coupons:
                            record.write(str(c) + "\n")
                        record.write("\n")
                    model.setCoupons([])
    # pool.close()
    # pool.join()
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