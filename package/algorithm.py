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
    def __init__(self, model:DiffusionModel, depth, cluster_theta=0.9, simulationTimes=1, num_sampledGraph=50):
        self._model = model
        self._graph = model.getGraph()
        self._reset_graph = None # Cluster
        self._itemset = model.getItemsetHandler()
        self._max_expected = dict()
        # self._limitNum = k
        self._num_sampledGraph = num_sampledGraph
        self._depth = depth
        self._cluster_theta = cluster_theta
        if type(self._graph) == SN_Graph:
            self._max_expected_len, self._max_expected_path = SN_Graph.max_product_path(self._graph, self._model.getSeeds())
        self.simulationTimes = simulationTimes

    def setGraph(self, graph):
        self._graph = graph
        self._model.setGraph(graph)

    def getGraph(self, ):
        return self._graph

    # def setLimitCoupon(self, k):
    #     self._limitNum = k

    def resetGraph(self):
        self.setGraph(copy.deepcopy(self._reset_graph))

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
                    for discount in np.arange(price_step ,disItemset.price, price_step):
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
                
                coupon = Coupon(item.price, item, item.price*discount, item)
                coupons.append(coupon)
        return coupons

    def genFullItemCoupons(self, starThreshold:float, step:float, discountPercentage:list[float]) -> list:
              
        coupons = []
        sumItemPrices = sum(self._itemset.PRICE.values())
        allItems = self._itemset[" ".join(self._itemset.PRICE.keys())]
        for account in np.arange(starThreshold, sumItemPrices, step):
        
            for p in discountPercentage:
                if p <= 0 or p > 1:
                    raise ValueError("The discount percentage of coupon should be in (0,1]")
                coupons.append(Coupon(account, allItems, account*p, allItems))
        
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
                    u,v,w = max(pre_edges, key=lambda x: x[2] )
                    self._model._propagate(u, v, predecessor_adopt[u])
                    mainItemset = user_proxy._adoptMainItemset(v)
                    if mainItemset:
                        revenue += mainItemset["items"].price
        
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
        self._model.diffusion(tagger, self._depth)

        self._model.setCoupons(coupons)
        self._model.setGraph(graph)
        return tagger["TagEstimatedRevenue"].amount()
    
    def genSelfCoupons(self):
        print("Clustering the graph...")
        start = time.time()

        cluster_subgraphs = []
        for i in range(self._num_sampledGraph):
            subgraph = self._graph.bfs_sampling(roots=self._model.getSeeds())
            cluster_graph = ClusterGraph(graph = subgraph, 
                                        seeds = self._model.getSeeds(),
                                        located = False,
                                        depth = self._depth,
                                        theta = self._cluster_theta)
            cluster_subgraphs.append(cluster_graph)

        print("Execution time for clustering graph: {}".format(time.time() - start))
        self.setGraph(cluster_graph)

        user_proxy = self._model.getUserProxy()
        coupons = []

        global_benfit = self._globally_estimate([])
        level_generators = [g.level_travesal(self._model.getSeeds(), self._depth) for g in cluster_subgraphs]
        current_level_superNodes = []
        # 所有子圖的種子節點
        for generator in level_generators:
            current_level_superNodes.append(next(generator))

        for i in range(self._depth):
            print("Level: {}".format(i))

            # 計算當前層數在沒有coupon的情況下，每張聚合子圖的平均收益
            max_local_benfit = 0
            next_level_superNodes = []
            for generator in level_generators:
                next_level_superNodes.append(next(generator))

            for subgraph in cluster_subgraphs:
                if i != self._depth:
                    max_local_benfit += self._locally_estimate(current_level_superNodes, next_level_superNodes)
                else:
                    max_local_benfit += self._locally_estimate(current_level_superNodes, [])
            max_local_benfit /= len(cluster_subgraphs)

            # TODO: 將相似的主商品聚合成一個super主商品，並且merge相似的super node變成super super node
            # BUG: 當我針對每個super node產生主商品時，有可能不在可以產生coupon的商品範圍內I_coupon
            # 將每個super node和相對應的主商品組成一組一組的tuple

            max_local_margin_coupon = None
            superNodeItemsetTuples = []
            for i in range(len(current_level_superNodes)):
                self.setGraph(cluster_subgraphs[i]) # NOTE: 檢查model有沒有儲存或異動跟圖狀態有關的變數(節點、邊之類的)
                super_nodes_same_graph = current_level_superNodes[i]
                for super_nodes in super_nodes_same_graph:
                    mainItemset = user_proxy._adoptMainItemset(super_nodes)
                    if not mainItemset: # TODO: 如果沒有商品在I_coupon裡面，可以直接跳過
                        continue
                    superNodeItemsetTuples.append(tuple(super_nodes, mainItemset))

                # TODO: clustering main itemset and then clustering super nodes
                superMainItemset = None
                ultraNode = None
                accItemset = superMainItemset

                # NOTE: 這裡是work around的方法。ultra node 是多個super node聚合在一起，所以不會存在在子圖裡，只能先透過新增節點的方式。
                self._graph.add_node(ultraNode, topic, desired_set=None, adopted_set=None)
                for disItemset in user_proxy.discoutableItems(ultraNode, accItemset):
                    discount = user_proxy._min_discount(ultraNode, accItemset, disItemset)
                    discount = math.ceil(discount*100)/100 # 無條件進位到小數點第二位
                    disItemset = self._itemset.difference(disItemset, accItemset)
                    coupon = Coupon(accItemset.price, accItemset, discount, disItemset)
                    
                    start = time.time()

                    # local 的評估
                    local_revenue = 0
                    for subgraph in cluster_subgraphs:
                        if i != self._depth:
                            local_revenue += self._locally_estimate(current_level_superNodes, next_level_superNodes, coupon)
                        else:
                            local_revenue += self._locally_estimate(current_level_superNodes, [], coupon)
                    local_revenue /= len(cluster_subgraphs)


                    # TODO:
                    # 一個cluster只取一張效益最高的coupon，並按照收益做排序
                    if local_revenue >= max_local_revenue:
                        max_local_revenue = local_revenue
                        max_local_margin_coupon = coupon

            start = time.time()
            if max_local_margin_coupon:
                global_margin_benfit = self._globally_estimate(max_local_margin_coupon)
                print("Global estimation for level: {}, {}".format(i, time.time()-start))
                if global_margin_benfit >= global_benfit:
                    global_benfit = global_margin_benfit
                    coupons.append(max_local_margin_coupon)
                    self._model.setCoupons(coupons)
                
                
        self._model.setCoupons([])
        return coupons
    
    def _parallel(self, index, coupons):
        
        print("{}: {}".format(time.ctime(), index))
        graph = copy.deepcopy(self._model.getGraph())
    
        model = DiffusionModel(graph, self._model.getItemsetHandler(), coupons, self._model.getThreshold())
    
        tagger = Tagger()
        tagger.setNext(TagRevenue())
        tagger.setNext(TagActiveNode())
        
        # bucket = dict()

        for _ in range(self.simulationTimes):
            model.resetGraph()
            model.diffusion(tagger)
        #     path = self._max_expected_path
        #     for node, p in path.items():
        #         src = p[0]
        #         if src not in bucket:
        #             bucket[src] = dict()
                
        #         itemset = str(model._graph.nodes[node]["adopted_set"])
        #         if itemset not in bucket[src]:
        #             bucket[src][itemset] = 1
        #         else:
        #             bucket[src][itemset] += 1
        # self._bucket = bucket 

        tagger["TagRevenue"].avg(self.simulationTimes)
        tagger["TagActiveNode"].avg(self.simulationTimes)

        return tagger
    
    def simulation(self, candidatedCoupons:list[Coupon]):
        '''
            simulation for hill-climbing like algorithm
        '''
        
        candidatedCoupons = candidatedCoupons[:]
        
         
        tagger = self._parallel(-1, [])
        yield [], tagger
        
        coupons = [(i, [candidatedCoupons[i]]) for i in range(len(candidatedCoupons))]
        revenue = tagger["TagRevenue"].expected_amount()
        output = [] # the coupon set which is maximum revenue

        with Pool(cpu_count()) as pool:
            while len(candidatedCoupons) != 0:
                '''
                    1. Simulate with all candidated coupon
                    2. Get the coupon which can maximize revenue, and delete it from the candidatings
                    3. Concatenate all of the candidateings with the maximize revenue coupon
                '''
                print("Number of candidated coupons: {}".format(len(coupons)))
                result = pool.starmap(self._parallel, coupons)

                maxMargin = 0
                maxIndex = 0
                    
                # find the maximum margin benfit of coupon
                for i in range(len(result)):
                    if result[i]["TagRevenue"].expected_amount() >= maxMargin:
                        maxMargin = result[i]["TagRevenue"].expected_amount()
                        maxIndex = i
                
                # if these coupons are more benfit than current coupons, add it and update 
                if maxMargin > revenue:
                    output = coupons[maxIndex][1]
                    revenue = maxMargin
                    tagger = result[maxIndex]
                    del candidatedCoupons[maxIndex]
                    coupons = [(i, coupons[maxIndex][1] + [candidatedCoupons[i]]) for i in range(len(candidatedCoupons))]
                    
                else:
                    break
            
                yield output, tagger
    
    def optimalAlgo(self, candidatedCoupons:list, num_coupons:int):

        # if num_coupons > self._limitNum or num_coupons > len(candidatedCoupons):
        #     raise ValueError("The size of coupon set should be less K or the number of candidated coupons.")
        
        pool = Pool(cpu_count())
        candidatedCoupons = candidatedCoupons[:]

        couponsPowerset = []
        i = 0
        for comb in combinations(candidatedCoupons, num_coupons):
            couponsPowerset.append((i, list(comb)))
            i += 1

        result = pool.starmap(self._parallel, couponsPowerset)
        
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

    # def simulated_annealing(self, candidatedCoupons, initial_temperature=10000, cooling_rate=2, number_inner_iter=1000, stopping_temperature=1000): # pragma: no cover

    #     current_solution = []
    #     current_temperature = initial_temperature
    #     # the key is the combination of index of candidatedCoupons with increasing, 
    #     # and the value is the reverse of result in whole diffusion.
    #     count_inner_iter = 0
    #     couponSzie = min(len(candidatedCoupons), self._limitNum)

    #     while current_temperature > stopping_temperature:

    #         new_solution = []
    #         while len(new_solution) <= couponSzie:
    #             new_solution = self._move(current_solution, candidatedCoupons)

    #         pool = Pool(cpu_count())
    #         args = [("current", current_solution, self._max_expected_len), ("new", new_solution, self._max_expected_len)]
    #         result = pool.map(self._parallel, args)
    #         pool.close()
    #         pool.join()

    #         # objective function(new_solution) - objective function(current_solution)
    #         delta = result[1]["TagRevenue"].expected_amount() - result[0]["TagRevenue"].expected_amount()
    #         tagger = result[0]

    #         if delta > 0:
    #             current_solution = new_solution
    #             tagger = result[1]
    #         else:
    #             acceptance_probability = math.exp(delta / current_temperature)
    #             if random.random() < acceptance_probability:
    #                 current_solution = new_solution
    #                 tagger = result[1]
                    
    #         count_inner_iter += 1
    #         if count_inner_iter > number_inner_iter:
    #             current_temperature -= cooling_rate
    #             count_inner_iter = 0

    #     return current_solution, tagger
    
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
