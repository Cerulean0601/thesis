
from itemset import ItemsetFlyweight
from social_graph import SN_Graph
from coupon import Coupon
from networkx import shortest_path_length
import pandas as pd
import logging

from tag import Tagger,  TagMainItemset, TagAppending

class Algorithm:
    def __init__(self, model):
        self._model = model
        self._graph = model.getGraph()
        self._itemset = model.getItemsetHandler()
        self._expected = dict()
        self._belonging = None

    def genAllCoupons(self, price_step:float):
        '''
            Generates a set of all possible coupons.
        '''
        coupons = []
        min_amount = min(self._itemset.PRICE.values())
        limit_amout = sum(self._itemset.PRICE.values())

        for threshold in range(min_amount, limit_amout, price_step):
            for accNumbering, accItemset in self._itemset:
                for discount in range(0 ,limit_amout, price_step):
                    for disNumbering, disItemset in self._itemset:
                        coupons.append(Coupon(threshold, accItemset, discount, disItemset))
    
        return coupons

    def _preporcessing(self):
        numSampling = 5
        subgraph = self._graph.sampling_subgraph(numSampling)
        #origin = self._model.getGraph()
        self._model.replaceGraph(subgraph)
        self._shortestPath(self._model.getSeeds())
        self._grouping()

        tagger = Tagger()
        tagger.setParams(belonging=self._belonging, expectedProbability=self._expected)
        tagger.setNext(TagMainItemset())
        tagger.setNext(TagAppending(self._model.getItemsetHandler()))

        self._model.diffusion(tagger)
        return tagger
    
    def _shortestPath(self, seeds):
        
        for seed in seeds:
            self._expected[seed] = shortest_path_length(self._graph, source=seed, weight="weight")
            self._expected[seed][seed] = 1

    def _grouping(self):
        seeds = self._model.getSeeds()
        if not self._expected:
            self._shortestPath(seeds)

        self._belonging = pd.DataFrame.from_dict(self._expected).idxmax(axis=1)
        for seed in seeds:
            self._belonging[seed] = seed
        logging.info("Grouping nodes have done.")
        logging.debug(self._belonging)

    def myselfAlgo(self):
        tagger = self._preporcessing()
        mainItemset = tagger.getNext()
        appending = mainItemset.getNext()
        print(mainItemset)
        print(appending)