import os
import unittest

# const.
DATA_ROOT = os.path.join("D:" + os.sep, "論文實驗", "data")
DBLP_PATH = os.path.join(DATA_ROOT, "dblp")
AMAZON_PATH = os.path.join(DATA_ROOT, "amazon")


def test():
    testRunner = unittest.TextTestRunner()
    suite = unittest.defaultTestLoader.discover("./test/")
    testRunner.run(suite)
    
def getItemsPrice() -> list:
    prices = dict()

    with open(AMAZON_PATH, "r", encoding="utr8") as dataFile:
        for line in dataFile:
            id, *context, price = line.split(",")
            prices[id] = price
    return prices

import sys
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
sys.path.append('D:\\論文實驗\\package')

from package.model import DiffusionModel
from package.topic import TopicModel
from package.social_graph import SN_Graph
from package.itemset import ItemsetFlyweight
from package.coupon import Coupon

# hyper param
NUM_TOPICS = 5
if __name__ == '__main__':
    
    test()
    
    # topicModel = TopicModel(DBLP_PATH + "\\topic_nodes.csv", AMAZON_PATH + "\\preprocessed_Sortware.csv", NUM_TOPICS)
    # topicModel.save()
    topicModel = TopicModel.load(NUM_TOPICS)

    graph = SN_Graph()
    graph.construct(DBLP_PATH + "\\edges", DBLP_PATH + "\\nodes", topicModel)

    prices = getItemsPrice()
    itemset = ItemsetFlyweight(prices, topicModel)

    coupons = [Coupon(20, "0763855553", 10, "0763855553")]
    model = DiffusionModel("dblp_amazon", graph, itemset, coupons)
    model.diffusion()
    model.save()