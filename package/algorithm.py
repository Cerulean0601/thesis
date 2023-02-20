
from itemset import ItemsetFlyweight
from social_graph import SN_Graph
from coupon import Coupon
from networkx import shortest_path_length
import pandas as pd
import logging
import heapq
from multiprocessing.pool import ThreadPool

from tag import *
from itemset import Itemset
from utils import dot

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
        self._model.setGraph(subgraph)
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
    
    def _getAccItemset(self, mainItemset, appending):
        
        if not appending:
            yield mainItemset
            return
        
        # sort powerset by price in descending order
        powerset = sorted([ obj for asin, obj in self._itemset.powerSet(appending)], 
                            key=lambda X: X.price)
        for unionMainItemset in powerset:
            yield self._itemset.union(unionMainItemset, mainItemset)

    def _getAccThreshold(self, mainItemset, accItemset):
        
        staringPrice = mainItemset.price
        
        diff = self._itemset.difference(accItemset, mainItemset)
        if not diff:
            yield staringPrice
        else:
            candidate = self._itemset.powerSet(diff)
            sortedCandidate = self._itemset.sortByPrice([obj for id, obj in candidate])
            for candidate in sortedCandidate:
                yield staringPrice + candidate.price

    def _sumGroupExpTopic(self, group):
        topic_size = len(self._graph.nodes[group]["topic"])
        groupingNodes = (node for node in self._belonging.keys() if self._belonging[node] == group)

        expectedTopic = [0]*topic_size
        for node in groupingNodes:
            for t in range(topic_size):
                expectedTopic[t] += self._graph.nodes[node]["topic"][t]*self._expected[group][node]
        
        return expectedTopic
        # topic = []
        # for t in zippedTopic:
        #     print(t)
        # return topic
    
    
    def _getDisItemset(self, expctedTopic, mainItemset, appending):
        
        threhold = dot(expctedTopic, mainItemset.topic)/mainItemset.price

        candidatedList = []
        for ids, obj in self._itemset.powerSet(appending):
            if threhold >= (dot(expctedTopic, obj.topic)/obj.price):
                candidatedList.append(obj)
        
        for obj in self._itemset.sortByPrice(candidatedList, reverse=True):
            yield obj
            

    def _getMinDiscount(self, expctedTopic:list, mainItemset:Itemset, accItemset:Itemset, accThreshold, disItemset:Itemset):

        tradeItemset = self._itemset.union(mainItemset, disItemset)

        accAmount = self._itemset.intersection(mainItemset, self._itemset[accItemset]) # 已累積的金額
        accAmount = accAmount.price if accAmount != None else 0
        mainProportion = min(mainItemset.price/accThreshold,1)

        term = dot(expctedTopic, tradeItemset.topic)*mainProportion
        term *= dot(expctedTopic, mainItemset.topic)/mainItemset.price
        term = tradeItemset.price - term

        if term > disItemset.price:
            logging.warn("The minimum discount is larger than the total account of discountable itmeset")
            return disItemset.price
        return term

    def genSelfCoupon(self):
        
        tagger = self._preporcessing()
        mainItemset = tagger.getNext()
        appending = mainItemset.getNext()
        seeds = self._model.getSeeds()

        result = []
        for seed in seeds:
            result.append(self._sumGroupExpTopic(seed))

        # pool = ThreadPool()
        # result = pool.imap(self._sumGroupExpTopic, [seed for seed in seeds])
        # pool.close()
        # pool.join()

        groupExpTopic = dict()

        for i in range(len(seeds)):
            groupExpTopic[seeds[i]] = result[i]

        coupons = []
        for group in self._model.getSeeds():
            maxExceptMain = self._itemset[mainItemset.maxExcepted(group)]
            maxAppendingID = appending.maxExcepted(group)
            if maxAppendingID == None:
                continue

            maxExceptAppending = self._itemset[maxAppendingID]

            for accItemset in self._getAccItemset(maxExceptMain, maxExceptAppending):
                for accThreshold in self._getAccThreshold(maxExceptMain, accItemset):
                    for disItemset in self._getDisItemset(groupExpTopic[group], maxExceptMain, maxExceptAppending):
                        discount = self._getMinDiscount(groupExpTopic[group], maxExceptMain, accItemset, accThreshold, disItemset)
                        coupons.append(Coupon(accThreshold, accItemset, discount, disItemset))
        
        return coupons
