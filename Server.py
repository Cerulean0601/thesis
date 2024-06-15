from flask import Flask, request, jsonify
import numpy as np
from os import cpu_count
from multiprocessing.pool import Pool

from package.tag import Tagger, TagRevenue, TagActiveNode
from package.model import DiffusionModel
from package.topic import TopicModel
from package.social_graph import SN_Graph
from package.itemset import ItemsetFlyweight, ItemRelation
from package.utils import getItemsPrice, read_items
from package.algorithm import Algorithm
from package.coupon import Coupon

app = Flask(__name__)

# CONSTANT
DATA_ROOT = "./data"
DBLP_PATH = DATA_ROOT + "/dblp"
AMAZON_PATH = DATA_ROOT + "/amazon"
FACEBOOK_PATH = DATA_ROOT + "/facebook"
CLUB_PATH = DATA_ROOT + "/Karate Club Network"
NOTIFY_ENDPOINT = r"https://notify.run/O6EfLmG6Tof1s5DljYB7"

NUM_TOPICS = 5

def str2Coupons(collection, itemsetHandler:ItemsetFlyweight):

    collection_objs = []
    for coupons in collection:
        objs = []
        for coupon in coupons:
            obj = Coupon(float(coupon["accThreshold"]), itemsetHandler[coupon["accItemset"]], float(coupon["discount"]), itemsetHandler[coupon["disItemset"]])
            objs.append(obj)
        collection_objs.append(objs)
    
    return collection_objs

def monte_carlo_simulation(algo:Algorithm, coupons_collection:list):

    model = algo._model
    results = []
    times = 10000

    with Pool() as pool:
        for coupons in coupons_collection:
            model.setCoupons(coupons)
            algo.simulationTimes = times//cpu_count()
            parallel_simulation_result = pool.starmap(algo._parallel, [(i, coupons) for i in range(cpu_count())])
            merge_result = {"revenue":0, "expected_revenue":0, "active_nodes":0, "expected_active_nodes":0}

            for result in parallel_simulation_result:
                merge_result["revenue"] += result["TagRevenue"]._amount
                merge_result["expected_revenue"] += result["TagRevenue"]._expected_amount
                merge_result["active_nodes"] += result["TagActiveNode"]._amount
                merge_result["expected_active_nodes"] += result["TagActiveNode"]._expected_amount
    
            for key, value in merge_result.items():
                merge_result[key] = value/cpu_count()

            results.append(merge_result)

    return results

@app.route('/monte-carlo', methods=['POST'])
def monte_carlo():
    '''
        參數:
        1. 滿額回饋
        2. 蒙地卡羅模擬的執行次數

        跑完蒙地卡羅模擬後，儲存滿額回饋方案以及相對應的收益，並且回傳全部的平均收益
    '''
    data = request.json

    GRAPH = CLUB_PATH
    topicModel = TopicModel(NUM_TOPICS)
    topicModel.read_topics(node_file=GRAPH + "/nodes_with_" + str(NUM_TOPICS) + "_topic.csv",
                           items_file=AMAZON_PATH + "/items_with_" + str(NUM_TOPICS) + "_topic.csv")

    graph = SN_Graph.construct(GRAPH + "/edges", topicModel, located=False)
    relation = ItemRelation()
    relation.construct(AMAZON_PATH + "/sample_items.csv")
    itemset = ItemsetFlyweight(getItemsPrice(AMAZON_PATH + "/sample_items.csv"), topicModel, relation)

    model = DiffusionModel(graph, itemset, threshold=10**(-5), name="amazon in dblp")
    seed_size = min(itemset.size, graph.number_of_nodes())
    seeds = model.selectSeeds(seed_size)
    model.allocate(seeds, [itemset[asin] for asin in itemset.PRICE.keys()])
    algo = Algorithm(model, k=20, depth=4, cluster_theta=0.4)

    coupons_data = data['coupons_collection']
    coupons_collection = str2Coupons(coupons_data, itemset)
    results = monte_carlo_simulation(algo, coupons_collection)

    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(debug=True)
