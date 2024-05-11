from networkx import set_node_attributes
import numpy as np
from multiprocessing import Pool
import copy 
from itertools import combinations
from os import cpu_count
import random, math
import time

from package.tag import *
from package.model import DiffusionModel
from package.social_graph import SN_Graph
from package.cluster_graph import ClusterGraph
from package.coupon import Coupon
from package.utils import dot

class Algorithm:
    def __init__(self, model, k, depth, cluster_theta=0.9, simulationTimes=1):
        self._model = model
        self._graph = model.getGraph()
        self._reset_graph = copy.deepcopy(self._graph)
        self._itemset = model.getItemsetHandler()
        self._max_expected = dict()
        self._limitNum = k
        self._depth = depth
        self._cluster_theta = cluster_theta
        if type(self._graph) == SN_Graph:
            self._max_expected_len, self._max_expected_path = SN_Graph.max_product_path(self._graph, self._model.getSeeds())
        self.simulationTimes = simulationTimes

    def setGraph(self, graph):
        self._graph = graph
        self._model.setGraph(graph)

    def setLimitCoupon(self, k):
        self._limitNum = k

    # def calculateMaxExpProbability(self):
    #     G = self._graph
    #     seeds = self._model.getSeeds()
    #     if seeds is None or len(seeds) == 0:
    #         raise ValueError("Seed set is empty.")
        
    #     self._max_expected_subgraph, self._max_expected = SN_Graph.compile_max_product_graph(G, seeds)

    def genAllCoupons(self, price_step:float):
        '''
            Generates a set of all possible coupons.
        '''
        coupons = []
        
        for accItemset in self._itemset:
            min_amount = min([self._itemset[numbering].price for numbering in accItemset.numbering])
            for threshold in np.arange(min_amount, accItemset.price + price_step - 1, price_step):
                for disItemset in self._itemset:
                    for discount in np.arange(5 ,disItemset.price, price_step):
                        coupons.append(Coupon(threshold, accItemset, discount, disItemset))
    
        return coupons
    
    def genDiscountItemCoupons(self, discounts: list) -> list[Coupon]:
        '''
            Generates a list of coupons that discount items with relative discount
            
            Args:
                discounts: A list of relative discount

            Returns:
                a list of coupons
        '''
        items = self._itemset.getSingleItems()
        coupons = list()

        for item in items:
            for discount in discounts:
                if discount <= 0 or discount > 1:
                    raise ValueError("the discount of coupon should be float type and positive.")
                
                coupon = Coupon(item.price, item, item.price*(1-discount), item)
                coupons.append(coupon)
        return coupons

    def genFullItemCoupons(self, starThreshold:float, step:float, dicountPercentage:float) -> list:
        if dicountPercentage <= 0 or dicountPercentage > 1:
            raise ValueError("The discount percentage of coupon should be in (0,1]")
        
        coupons = []
        sumItemPrices = sum(self._itemset.PRICE.values())
        allItems = self._itemset[" ".join(self._itemset.PRICE.keys())]
        for account in np.arange(starThreshold, sumItemPrices, step):
            coupons.append(Coupon(account, allItems, account*dicountPercentage, allItems))
        
        return coupons
    def _locally_estimate(self, clusters:list, post_cluster:list, coupon:Coupon = None) -> float:
        user_proxy = self._model.getUserProxy()
        coupons = user_proxy.getCoupons()

        if not isinstance(user_proxy._graph, ClusterGraph) and type(self._model.getGraph()) != type(user_proxy._graph):
            raise TypeError("The type of graph should be ClusterGraph.")

        graph = user_proxy._graph
        if coupon:
            user_proxy.setCoupons(coupons + [coupon])
        revenue = 0
        predecessor_adopt = dict()

        for cluster in clusters:
            cluster_data = copy.deepcopy(user_proxy._graph.nodes[cluster])
            adopted_result = user_proxy.adopt(cluster)
            nx.set_node_attributes(graph, {cluster:cluster_data})
            if adopted_result:
                predecessor_adopt[cluster] = adopted_result["decision_items"]
                revenue += adopted_result["amount"]
        
        if clusters != post_cluster:
            for cluster in post_cluster:
                pre_edges = list(filter(lambda x: (x[0] in clusters) and (x[0] in predecessor_adopt), graph.in_edges(nbunch=cluster, data="weight")))
                if pre_edges:
                    u,v,w = min(pre_edges, key=lambda x: x[2] )
                    self._model._propagate(u, v, predecessor_adopt[u])
                    mainItemset = user_proxy._adoptMainItemset(v)
                    if mainItemset:
                        revenue += w*mainItemset["items"].price
        
        user_proxy.setCoupons(coupons)
        return revenue
    
    def _globally_estimate(self, coupon:Coupon = None) -> float:
        graph = self._model.getGraph()
        if not isinstance(graph, ClusterGraph):
            raise TypeError("The type of graph should be ClusterGraph.")
        
        coupons = self._model.getCoupons()
        if coupon:
            self._model.setCoupons(coupons + [coupon])
        self._model.setGraph(copy.deepcopy(self._reset_graph))

        tagger = Tagger()
        tagger.setNext(TagEstimatedRevenue(graph=self._model.getGraph()))
        self._model.DeterministicDiffusion(self._depth, tagger)

        self._model.setCoupons(coupons)
        self._model.setGraph(graph)
        return tagger["TagEstimatedRevenue"].amount()
    def genSelfCoupons(self):
        print("Clustering the graph...")
        start = time.time()
        cluster_graph = ClusterGraph(graph = self._graph, 
                                    seeds = self._model.getSeeds(),
                                    located = False,
                                    depth = self._depth,
                                    theta = self._cluster_theta)
        print("Execution time for clustering graph: {}".format(time.time() - start))

        self._model.setGraph(cluster_graph)
        self._reset_graph = copy.deepcopy(cluster_graph)

        user_proxy = self._model.getUserProxy()
        coupons = []

        level_clusters = list(cluster_graph._level_travesal(self._model.getSeeds(), self._depth))
        leaf_level = len(level_clusters)-1
        global_benfit = self._globally_estimate([])
        for i in range(leaf_level+1):
            print("Level: {}".format(i))

            if i != leaf_level:
                max_local_benfit = self._locally_estimate(level_clusters[i], level_clusters[min(i+1, leaf_level)])
            else:
                max_local_benfit = self._locally_estimate(level_clusters[i], [])
            max_local_margin_coupon = None

            for cluster in level_clusters[i]:
                mainItemset = user_proxy._adoptMainItemset(cluster)
                if not mainItemset: 
                    continue

                accItemset = mainItemset["items"]
                for disItemset in user_proxy.discoutableItems(cluster, accItemset):
                    discount = user_proxy._min_discount(cluster, accItemset, disItemset)
                    discount = math.ceil(discount*100)/100 # 無條件進位到小數點第二位
                    disItemset = self._itemset.difference(disItemset, accItemset)
                    coupon = Coupon(accItemset.price, accItemset, discount, disItemset)
                    
                    start = time.time()
                    if i != leaf_level:
                        margin_benfit = self._locally_estimate(level_clusters[i], level_clusters[min(i+1, leaf_level)], coupon)
                    else:
                        margin_benfit = self._locally_estimate(level_clusters[i], [], coupon)
                    # print("Local estimation for level {}: {}".format(i, time.time()-start))
                    if margin_benfit > max_local_benfit:
                        max_local_benfit = margin_benfit
                        max_local_margin_coupon = coupon

            start = time.time()
            if max_local_margin_coupon:
                global_margin_benfit = self._globally_estimate(max_local_margin_coupon)
                print("Global estimation for level: {}, {}".format(i, time.time()-start))
                if global_margin_benfit > global_benfit:
                    global_benfit = global_margin_benfit
                    coupons.append(coupon)
                    self._model.setCoupons(coupons)
        self._model.setGraph(self._graph)
        return coupons
    
    def _parallel(self, args):
        coupon = args[1]
        graph = copy.deepcopy(self._model.getGraph())
    
        model = DiffusionModel(graph, self._model.getItemsetHandler(), coupon, self._model.getThreshold())
    
        tagger = Tagger()
        tagger.setNext(TagRevenue(graph, self._model.getSeeds()))
        tagger.setNext(TagActiveNode())
        
        bucket = dict()

        for time in range(self.simulationTimes):
            # initialize for Monte Carlo Simulation
            graph.initAttr()
            seeds = self._model.getSeeds()
            if seeds:
                model.setSeeds(seeds)
                data = dict()
                for seed in seeds:
                    data[seed] = copy.deepcopy(self._model.getGraph().nodes[seed])
                    data[seed]["adopted_records"] = list()
                set_node_attributes(model.getGraph(), data)
            
            model.diffusion(tagger)

            path = self._max_expected_path
            for node, p in path.items():
                src = p[0]
                if src not in bucket:
                    bucket[src] = dict()
                
                itemset = str(model._graph.nodes[node]["adopted_set"])
                if itemset not in bucket[src]:
                    bucket[src][itemset] = 1
                else:
                    bucket[src][itemset] += 1
        self._bucket = bucket 

        tagger["TagRevenue"].avg(self.simulationTimes)
        tagger["TagActiveNode"].avg(self.simulationTimes)

        return tagger
    
    def simulation(self, candidatedCoupons:list[Coupon]):
        '''
            simulation for hill-climbing like algorithm
        '''
        
        candidatedCoupons = candidatedCoupons[:]

        if len(candidatedCoupons) == 0:
            return [], self._parallel((0,[]))
        
        coupons = [(i, [candidatedCoupons[i]]) for i in range(len(candidatedCoupons))]
        output = [] # the coupon set which is maximum revenue 
        revenue = 0
        tagger = None

        while len(candidatedCoupons) != 0 and len(output) < self._limitNum:
            '''
                1. Simulate with all candidated coupon
                2. Get the coupon which can maximize revenue, and delete it from the candidatings
                3. Concatenate all of the candidateings with the maximize revenue coupon
            '''

            pool = Pool(cpu_count())
            result = pool.map(self._parallel, coupons)
            pool.close()
            pool.join()

            maxMargin = 0
            maxIndex = 0
                
            # find the maximum margin benfit of coupon
            for i in range(len(result)):
                if result[i]["TagRevenue"].expected_amount() > maxMargin:
                    maxMargin = result[i]["TagRevenue"].expected_amount()
                    maxIndex = i
                elif result[i]["TagRevenue"].expected_amount() == maxMargin:
                    if result[i]["TagActiveNode"].expected_amount() > result[maxIndex]["TagActiveNode"].expected_amount():
                        maxIndex = i
            
            # if these coupons are more benfit than current coupons, add it and update 
            if maxMargin > revenue:
                output = coupons[maxIndex][1]
                revenue = maxMargin
                tagger = result[maxIndex]
                del candidatedCoupons[maxIndex]
                coupons = [(i, coupons[maxIndex][1] + [candidatedCoupons[i]], self._max_expected_len) for i in range(len(candidatedCoupons))]
                
            else:
                break
            
        return output, tagger
    
    def optimalAlgo(self, candidatedCoupons:list):

        pool = Pool(cpu_count())
        candidatedCoupons = candidatedCoupons[:]
        couponSize = min(len(candidatedCoupons), self._limitNum)

        couponsPowerset = []
        i = 0
        for size in range(couponSize + 1): 
            for comb in combinations(candidatedCoupons, size):
                couponsPowerset.append((i, list(comb), self._max_expected_len))
                i += 1

        result = pool.map(self._parallel, couponsPowerset)
        
        pool.close()
        pool.join()
        
        maxRevenue = 0
        maxIndex = 0

        for i in range(len(result)):
            if result[i]["TagRevenue"].expected_amount() > maxRevenue:
                maxRevenue = result[i]["TagRevenue"].expected_amount()
                maxIndex = i

        return couponsPowerset[maxIndex][1], result[maxIndex]
    
    def _move(self, current_solution, candidated): # pragma: no cover
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

    def simulated_annealing(self, candidatedCoupons, initial_temperature=10000, cooling_rate=2, number_inner_iter=1000, stopping_temperature=1000): # pragma: no cover

        current_solution = []
        current_temperature = initial_temperature
        # the key is the combination of index of candidatedCoupons with increasing, 
        # and the value is the reverse of result in whole diffusion.
        count_inner_iter = 0
        couponSzie = min(len(candidatedCoupons), self._limitNum)

        while current_temperature > stopping_temperature:

            new_solution = []
            while len(new_solution) <= couponSzie:
                new_solution = self._move(current_solution, candidatedCoupons)

            pool = Pool(cpu_count())
            args = [("current", current_solution, self._max_expected_len), ("new", new_solution, self._max_expected_len)]
            result = pool.map(self._parallel, args)
            pool.close()
            pool.join()

            # objective function(new_solution) - objective function(current_solution)
            delta = result[1]["TagRevenue"].expected_amount() - result[0]["TagRevenue"].expected_amount()
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
        
#         pool = Pool(cpu_count())
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
