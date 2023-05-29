
from networkx import set_node_attributes, get_node_attributes
import pandas as pd
import logging
import numpy as np
from multiprocessing.pool import ThreadPool
import multiprocessing
import copy 
from itertools import combinations
from os import cpu_count
import random, math
from operator import add

from package.tag import *
from package.itemset import Itemset
from package.model import DiffusionModel
from package.itemset import ItemsetFlyweight
from package.social_graph import SN_Graph
from package.coupon import Coupon

class Algorithm:
    def __init__(self, model, k):
        self._model = model
        self._graph = model.getGraph()
        self._itemset = model.getItemsetHandler()
        self._max_expected = dict()
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
            raise ValueError("Should 2wsaZ the seeds of model before preprocessing.")
        
        subgraph = self._graph.sampling_subgraph(numSampling, roots=self._model.getSeeds())
        
        # The subgraph has been sampled with deepcopy
        # sub_model = DiffusionModel("Subgraph", 
        #                            subgraph,
        #                            self._model.getItemsetHandler(),
        #                            self._model.getCoupons(),
        #                            self._model.getThreshold()
        #                            )
        # sub_model._seeds = copy.deepcopy(self._model.getSeeds())
        # sub_model.allocate(sub_model._seeds, [self._itemset[id] for id in list(self._itemset.PRICE.keys())])
        sub_model = copy.deepcopy(self._model)
        
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
        normExpected = dict()
        for group, expectedGroup in self._expected.items():
                normExpected[group] = sum(expectedGroup.values())

        for node in groupingNodes:
            for t in range(topic_size):
                expectedTopic[t] += self._graph.nodes[node]["topic"][t]*(self._expected[group][node]/normExpected[group])
        
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
            maxExceptMainID = mainItemset.maxExcepted(group)
            maxAppendingID = appending.maxExcepted(group)
            if not maxAppendingID or not maxExceptMainID:
                raise ValueError("The main itemset or appendings of group {0} is empty".format(group))

            maxExceptMain = self._itemset[maxExceptMainID]
            maxExceptAppending = self._itemset[maxAppendingID]

            for accItemset in self._getAccItemset(maxExceptMain, maxExceptAppending):
                for accThreshold in self._getAccThreshold(maxExceptMain, accItemset):
                    for disItemset in self._getDisItemset(groupExpTopic[group], maxExceptMain, maxExceptAppending):
                        discount = self._getMinDiscount(groupExpTopic[group], maxExceptMain, accItemset, accThreshold, disItemset)
                        coupons.append(Coupon(accThreshold, accItemset, discount, disItemset))
        
        return coupons
    
    def genModifiedGreedyCoupon(self, price_step):
        
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
                    all_items = " ".join(list(self._itemset.PRICE.keys())) 
                    disItemset = self._itemset.difference(self._itemset[all_items], maxExceptMain)
                    for discount in np.arange(5 ,disItemset.price, price_step):
                        coupons.append(Coupon(accThreshold, accItemset, discount, disItemset))
        
        return coupons
    
    def dynamicProgramming(self, candidatedCoupons):
        coupon_set = []
        limit_size = min(self._limitNum, len(candidatedCoupons))
        for size in range(limit_size + 1):
            for comb_coupos in combinations(candidatedCoupons, size):
                # comb_coupos is tuple type
                coupon_set.append(list(comb_coupos))

        '''
        init CP, adopt table for users where the size of coupon set is one, and caculate revenue
        row is desired set of a user, and column is the coupon

        if the size of coupon set is more than one, CP(D, C)=max(CP(D,c1),CP(D,c2)
        '''
    def _parallel(self, args):
        coupon = args[1]
        graph = copy.deepcopy(self._model.getGraph())
        model = DiffusionModel("", graph, self._model.getItemsetHandler(), coupon, self._model.getThreshold())
        
        seeds = self._model.getSeeds()
        if seeds != None and len(seeds) > 0:
            model.setSeeds(seeds)
            for seed in seeds:
                data = {seed: self._model.getGraph().nodes[seed]}
                set_node_attributes(model.getGraph(), data)
        tagger = Tagger()
        tagger.setNext(TagRevenue(graph, model.getSeeds(), args[2]))
        tagger.setNext(TagActiveNode())
        model.diffusion(tagger)
        # count = [0]*5
        # for node, attr in graph.nodes(data=True):
        #     l = len(attr["adopted_records"])
        #     if len(count) < l:
        #         count.extend([0]*(l-len(count) + 1))
        #     count[l] += 1
        # print(count)
        return tagger
    
    def simulation(self, candidatedCoupons:list[Coupon]):
        '''
            simulation for hill-climbing like algorithm
        '''
        
        candidatedCoupons = candidatedCoupons[:]
        max_expected_path = dict()

        if len(candidatedCoupons) == 0:
            return [], self._parallel((0,[],max_expected_path))
        
        coupons = [(i, [candidatedCoupons[i]], max_expected_path) for i in range(len(candidatedCoupons))]
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
                coupons = [(i, coupons[maxIndex][1] + [candidatedCoupons[i]], max_expected_path) for i in range(len(candidatedCoupons))]
                
            else:
                break
            
        return output, tagger
    
    def optimalAlgo(self, candidatedCoupons:list):

        pool = ThreadPool(cpu_count())
        candidatedCoupons = candidatedCoupons[:]
        couponSzie = min(len(candidatedCoupons), self._limitNum)

        couponsPowerset = []
        i = 0
        max_expected_path = dict()
        for size in range(couponSzie + 1): 
            for comb in combinations(candidatedCoupons, size):
                couponsPowerset.append((i, list(comb), max_expected_path))
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
    
    def _move(self, current_solution, candidated):
        neighbors_solution = current_solution[:]

        p = random.uniform(0, 1)
        if p < 1/3:
            # remove coupon from current solution
            pop_index = random.randint(0, len(neighbors_solution)-1)
            neighbors_solution.pop(pop_index)

        elif p >= 1/3 and p < 2/3:
            # swap coupon
            while(True):
                replace_index = random.randint(0, len(neighbors_solution)-1)
                from_index = random.randint(0, len(candidated)-1)
                if candidated[from_index] not in neighbors_solution:
                    neighbors_solution[replace_index] = candidated[from_index]
                    break
        else:
            # add coupon to current solution
            while(True):
                add_index = random.randint(0, len(neighbors_solution)-1)
                if candidated[add_index] not in neighbors_solution:
                    neighbors_solution.append(candidated[add_index])
                    break

        return neighbors_solution

    def simulated_annealing(self, candidatedCoupons, initial_temperature=10000, cooling_rate=2, number_inner_iter=1000, stopping_temperature=1000):

        current_solution = []
        current_temperature = initial_temperature
        # the key is the combination of index of candidatedCoupons with increasing, 
        # and the value is the reverse of result in whole diffusion.
        count_inner_iter = 0
        couponSzie = min(len(candidatedCoupons), self._limitNum)
        max_expected_path = dict()

        while current_temperature > stopping_temperature:

            new_solution = []
            while len(new_solution) <= couponSzie:
                new_solution = self._move(current_solution, candidatedCoupons)

            pool = ThreadPool(cpu_count())
            args = [("current", current_solution, max_expected_path), ("new", new_solution, max_expected_path)]
            result = pool.map(self._parallel, args)
            pool.close()
            pool.join()

            # objective function(new_solution) - objective function(current_solution)
            delta = result[1]["TagRevenue"].amount() - result[0]["TagRevenue"].amount()
            tagger = result[0]

            if delta > 0:
                current_solution = new_solution
                tagger = result[1]
            else:
                acceptance_probability = math.exp(delta / current_temperature)
                if random.random() < acceptance_probability:
                    current_solution = new_solution
                    tagger = result[1]
                    
            count_inner_iter += 1
            if count_inner_iter > number_inner_iter:
                current_temperature -= cooling_rate
                count_inner_iter = 0

        return current_solution, tagger
    
# class DynamicProgram:
#     def __init__(self, model, k):
#         '''
#         NOTE: This class will modify the graph of model. 
#         If you don't want to modify, should be deep copy before passing it
#         '''
#         self._model = model
#         self._limitNum = k
#         self._max_expected = dict()
    
#     def setLimitCoupon(self, k):
#         self._limitNum = k

#     def compile_max_product_graph(self) -> SN_Graph:
#         seeds = self._model.getSeeds()
#         if seeds is None or len(seeds) == 0:
#             raise ValueError("The seed set is empty!")
        
#         tree_graph = SN_Graph(node_topic=self._model.getGraph().topic, located=False)
#         length, path = max_product_path(self._model.getGraph(), seeds)

#         for node, halfway in path.items():
#             if len(halfway) == 1:
#                 tree_graph.add_node(node)
#             for i in range(len(halfway)-1, 0, -1):
#                 # 從後面的節點開始連邊，如果邊已經連過了就跳出迴圈
#                 if not tree_graph.has_edge(halfway[i - 1], halfway[i]):
#                     weight = length[halfway[i]] / length[halfway[i - 1]]
#                     tree_graph.add_edge(halfway[i - 1], halfway[i], weight=weight)

#         tree_graph._initAllNodes()
#         self._max_expected = length

#         return tree_graph

#     def adopt_all_coupons(self, node, desired_set_list, coupons_powerset):
#         # return the adopted set in different coupon set

#         def _parallel(args):
#             '''
#             Param:
#                 vp_ratio (dict[list]): vp ratio of combination of desired_set and a coupon
#                 adopt_itemset (dict[list]): adopt itemset of a user in differenct desired set and a coupon
#                 index (int): the index of coupons_powerset
            
#             Returns:
#                 adopt_itemset (Itemset), expected_amount(float)
#             '''
#             vp_ratio = args[0]
#             index = args[1]
#             desired_set = desired_set_list[index]
#             coupon_set = coupons_powerset[index]

#             max_index = -1
#             max_vp = 0
#             for num_coupon in coupon_set:
#                 if max_vp < vp_ratio[str(desired_set)][num_coupon]:
#                     max_vp = vp_ratio[str(desired_set)][num_coupon]
#                     max_index = num_coupon

#             return max_index

#         user_proxy = self._model.getUserProxy()
#         coupons = self._model.getCoupons()
#         graph = self._model.getGraph()

#         vp_table = dict()
#         amount = []
#         adopt_itemset = []

#         distinct_desired_set = set(desired_set_list)
#         for desired_set in distinct_desired_set:
#             vp_table[str(desired_set)] = [0]*len(self._model.getCoupons()) 

#             for i in range(len(coupons_powerset)):
#                 # for single coupon
#                 if len(coupons_powerset[i]) > 1:
#                     break
#                 else:
#                     user_proxy.setCoupons([coupons[coupons_powerset[i][0]]])
#                     graph.nodes[node]["desired_set"] = desired_set
#                     result = user_proxy.adopt(node)
#                     vp_table[str(desired_set)][i] = result["VP"]
#                     amount.append(result["amount"]*self._max_expected[node])
#                     adopt_itemset.append(result["tradeOff_items"])

#                     # reset the set of user
#                     graph.nodes[node]["desired_set"] = None
#                     graph.nodes[node]["adopted_set"] = None
#                     user_proxy.setCoupons(coupons)
        
#         pool = ThreadPool(cpu_count())
#         args = [(vp_table, i) for i in range(len(coupons), len(coupons_powerset))]
#         result = pool.map(_parallel, args)
#         for index in result:
#             amount.append(amount[index]*self._max_expected[node])
#             adopt_itemset.append(adopt_itemset[index])
        
#         return adopt_itemset, amount
    
#     def run(self):
#         coupons = self._model.getCoupons()
        
#         encode_coupons = list(range(len(coupons)))
#         num_coupons = min(self._limitNum, len(encode_coupons))
#         coupons_powerset = []
#         # at least one coupon
#         for size in range(1, num_coupons+1):
#             for num_set in combinations(encode_coupons, size):
#                 coupons_powerset.append(num_set)
        
#         seeds = self._model.getSeeds()
#         # copy desired_set of seeds
#         origin_g = self._model.getGraph()
#         graph = self.compile_max_product_graph()

#         for seed in seeds:
#             graph.nodes[seed]["desired_set"] = origin_g.nodes[seed]["desired_set"]
#         self._model.setGraph(graph)
        
#         revenue = [0] * len(coupons_powerset) # the index map to the index of coupons_powerset
#         bfs_queue = []
#         for s in seeds:
#             # node, desired_set, index of coupons_powerset
#             desired_set = [self._model.getGraph().nodes[s]["desired_set"]] * len(coupons_powerset)
#             bfs_queue.append([s, desired_set])

#         while len(bfs_queue) > 0:
#             task = bfs_queue.pop(0)
#             node, desired_set = task[0], task[1]
#             adopt_itemset, amount = self.adopt_all_coupons(node, desired_set, coupons_powerset)

#             for out_neighbor in graph.neighbors(node):
#                 bfs_queue.append([out_neighbor, adopt_itemset])

#             revenue = list(map(add, amount, revenue))

#         print(revenue)
#         return [coupons[i] for i in coupons_powerset[revenue.index(max(revenue))]]
