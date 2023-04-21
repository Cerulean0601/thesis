
from itemset import ItemsetFlyweight
from social_graph import SN_Graph
from coupon import Coupon
from networkx import shortest_path_length
import pandas as pd
import logging
import numpy as np
from multiprocessing.pool import ThreadPool
import multiprocessing
import copy 
from itertools import combinations
from os import cpu_count

from tag import *
from itemset import Itemset
from utils import dot
from model import DiffusionModel

class Algorithm:
    def __init__(self, model, k):
        self._model = model
        self._graph = model.getGraph()
        self._itemset = model.getItemsetHandler()
        self._expected = dict()
        self._belonging = None
        self._limitNum = k

    def setLimitCoupon(self, k):
        self._limitNum = k

    def genAllCoupons(self, price_step:float):
        '''
            Generates a set of all possible coupons.
        '''
        coupons = []
        
        for accNumbering, accItemset in self._itemset:
            min_amount = min([self._itemset[numbering].price for numbering in accItemset.numbering])
            for threshold in np.arange(min_amount, accItemset.price + price_step - 1, price_step):
                for disNumbering, disItemset in self._itemset:
                    for discount in np.arange(5 ,disItemset.price, price_step):
                        coupons.append(Coupon(threshold, accItemset, discount, disItemset))
    
        return coupons

    def _preprocessing(self):
        numSampling = 5
        if len(self._model.getSeeds()) == 0:
            raise ValueError("Should select the seeds of model before preprocessing.")
        
        subgraph = self._graph.sampling_subgraph(numSampling, roots=self._model.getSeeds())
        
        # The subgraph has been sampled with deepcopy
        sub_model = DiffusionModel("Subgraph", 
                                   subgraph, 
                                   self._model.getItemsetHandler(),
                                   self._model.getCoupons(),
                                   self._model.getThreshold()
                                   )
        
        sub_model._seeds = copy.deepcopy(self._model.getSeeds())
        sub_model.allocate(sub_model._seeds, [self._itemset[id] for id in list(self._itemset.PRICE.keys())])

        
        self._shortestPath(sub_model.getSeeds())
        self._grouping(sub_model)

        tagger = Tagger()
        tagger.setParams(belonging=self._belonging, expectedProbability=self._expected)
        tagger.setNext(TagMainItemset())
        tagger.setNext(TagAppending(sub_model.getItemsetHandler()))
        sub_model.diffusion(tagger)

        return tagger
    
    def _shortestPath(self, seeds):
        
        for seed in seeds:
            self._expected[seed] = shortest_path_length(self._graph, source=seed, weight="weight")
            self._expected[seed][seed] = 1

    def _grouping(self, model):
        seeds = model.getSeeds()
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

    def genSelfCoupons(self):
        
        # select seeds if the set is empty
        seeds = self._model.getSeeds()
        if not seeds:
            k = min(self._model._itemset.size, self._model._graph.number_of_nodes())

            # list of the seeds is sorted by out-degree.
            self._model.selectSeeds(k)
            seeds = self._model.getSeeds()

        if self._model._graph.nodes[seeds[0]]["desired_set"] == None:
            # List of single items.
            items = [self._model._itemset[id] for id in list(self._model._itemset.PRICE.keys())]

            self._model.allocate(seeds, items)

        tagger = self._preprocessing()
        mainItemset = tagger.getNext()
        appending = mainItemset.getNext()
        

        # result = []
        # for seed in seeds:
        #     result.append(self._sumGroupExpTopic(seed))

        pool = ThreadPool()
        result = pool.map(self._sumGroupExpTopic, [seed for seed in seeds])
        pool.close()
        pool.join()

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
    
    def _parallel(self, args):
        coupon = args[1]
        graph = self._model.getGraph()
        model = DiffusionModel("", copy.deepcopy(graph), self._model.getItemsetHandler(), coupon, self._model.getThreshold())
        tagger = Tagger()
        tagger.setNext(TagRevenue(graph, args[2]))
        tagger.setNext(TagActivatedNode())
        model.diffusion(tagger)

        return tagger
    
    def simulation(self, candidatedCoupons):
        '''
            simulation for hill-climbing like algorithm
        '''
        

        candidatedCoupons = candidatedCoupons[:]
        shortest_path_length = dict()

        if len(candidatedCoupons) == 0:
            return [], self._parallel((0,[],shortest_path_length))
        
        coupons = [(i, [candidatedCoupons[i]], shortest_path_length) for i in range(len(candidatedCoupons))]
        output = [] # the coupon set which is maximum revenue 
        revenue = 0
        tagger = None

        while len(candidatedCoupons) != 0 and len(output) < self._limitNum:
            '''
                1. Simulate with all candidated coupon
                2. Get the coupon which can maximize revenue, and delete it from the candidatings
                3. Concatenate all of the candidateings with the maximize revenue coupon
            '''

            pool = ThreadPool(cpu_count())
            result = pool.map(self._parallel, coupons)
            pool.close()
            pool.join()

            maxMargin = 0
            maxIndex = 0
                
            # find the maximum margin benfit of coupon
            for i in range(len(result)):
                if result[i]["TagRevenue"].amount() > maxMargin:
                    maxMargin = result[i]["TagRevenue"].amount()
                    maxIndex = i
            
            # if these coupons are more benfit than current coupons, add it and update 
            if maxMargin > revenue:
                output = coupons[maxIndex][1]
                revenue = maxMargin
                tagger = result[maxIndex]
                del candidatedCoupons[maxIndex]
                coupons = [(i, coupons[maxIndex][1] + [candidatedCoupons[i]], shortest_path_length) for i in range(len(candidatedCoupons))]
                
            else:
                break
            
        return output, tagger
    
    def optimalAlgo(self, candidatedCoupons:list):

        pool = ThreadPool(cpu_count())
        candidatedCoupons = candidatedCoupons[:]
        couponSzie = min(len(candidatedCoupons), self._limitNum)

        couponsPowerset = []
        i = 0
        shortest_path_length = dict()
        for size in range(couponSzie + 1): 
            for comb in combinations(candidatedCoupons, size):
                couponsPowerset.append((i, list(comb), shortest_path_length))
                i += 1

        result = pool.map(self._parallel, couponsPowerset)
        
        pool.close()
        pool.join()
        
        maxRevenue = 0
        maxIndex = 0

        for i in range(len(result)):
            if result[i]["TagRevenue"].amount() > maxRevenue:
                maxRevenue = result[i]["TagRevenue"].amount()
                maxIndex = i

        return couponsPowerset[maxIndex][1], result[maxIndex]

        