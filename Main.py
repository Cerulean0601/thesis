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

import sys
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
sys.path.append('D:\\論文實驗\\package')

from model import DiffusionModel
from topic import TopicModel
from social_graph import SN_Graph
from itemset import ItemsetFlyweight, ItemRelation
from coupon import Coupon
from utils import getItemsPrice
from algorithm import Algorithm
# hyper param
NUM_TOPICS = 5
if __name__ == '__main__':
    
    # test()
    NODE_TOKEN_FILE = os.path.join(DBLP_PATH, "token_nodes.csv")
    NODE_FILE, EDGE_FILE = os.path.join(DBLP_PATH, "nodes"), os.path.join(DBLP_PATH, "edges")
    ITME_FILE = os.path.join(AMAZON_PATH, "preprocessed_Software.csv")

    # topicModel = TopicModel(NUM_TOPICS)
    # topicModel.construct(NODE_TOKEN_FILE, ITME_FILE)
    # topicModel.save()
    topicModel = TopicModel.load(NUM_TOPICS)

    graph = SN_Graph()
    graph.construct(NODE_FILE, EDGE_FILE, topicModel.getNodesTopic())

    prices = getItemsPrice(ITME_FILE)
    relation = ItemRelation()
    relation.construct(ITME_FILE)
    itemset = ItemsetFlyweight(prices, topicModel.getItemsTopic(), relation)

    alog = Algorithm(graph, itemset)
    coupons = alog.greedy(5)
    model = DiffusionModel("dblp_amazon", graph, itemset, coupons)
    model.diffusion()
    model.save()

    
    

