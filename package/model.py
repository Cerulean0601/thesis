from queue import Queue
from random import uniform

# custom package
from package.user_proxy import UsersProxy
from package.itemset import ItemsetFlyweight, Itemset
from package.coupon import Coupon
from package.social_graph import SN_Graph
from package.cluster_graph import ClusterGraph
#from tag import Tagger, TagMainItemset, TagAppending

class DiffusionModel():
    def __init__(self, graph:SN_Graph|ClusterGraph, itemset, coupons=[], threshold = 0.0, name=None) -> None:
        self._graph = graph
        self.name = name

        self._itemset = ItemsetFlyweight(itemset["price"], itemset["topic"]) if type(itemset) == dict else itemset
        self._user_proxy = UsersProxy(self._graph, self._itemset, coupons, threshold)
        self._seeds = []
        self.influencedNodes = []
        
    def getSeeds(self):
        return self._seeds

    def setSeeds(self, seeds):
        self._seeds = seeds
        
    def getGraph(self):
        return self._graph

    def setGraph(self, newGraph:SN_Graph|ClusterGraph):
        if not isinstance(newGraph, (ClusterGraph, SN_Graph)):
            raise TypeError("Replaced graph is not SN_Graph or ClusterGraph class.\n")
        
        self._user_proxy.setGraph(newGraph)
        self._graph = newGraph
    
    def resetGraph(self):
        graph = self._graph
        for u in self.influencedNodes:
            graph._initNode(u)
            for v in graph.neighbors(u):
                graph._initEdge(u, v)

    def getItemsetHandler(self):
        return self._itemset

    def getUserProxy(self):
        return self._user_proxy
    
    def setCoupons(self, coupons:list[Coupon]):
        self.getUserProxy().setCoupons(coupons)

    def getCoupons(self):
        return self.getUserProxy().getCoupons()
    
    def getThreshold(self):
        return self._user_proxy.getThreshold()

    def selectSeeds(self, k:int) -> list:
        '''
            Return:
                list: an ordered list of nodes which is sorted by degrees
        '''
        self._seeds = [seed[0] for seed in self._graph.top_k_nodes(k)]
        return self._seeds
    
    def allocate(self, seeds, items):
        '''
        將商品分配給種子節點，並且放進對應的 desired_set。
        分配策略: 單價越高的商品分給degree越高的節點
        Args:
            seeds (list): 
                A list of seeds
            items (list of Itemset): 
                A list of items
        '''
        for i in range(len(items)):
            for j in range(i):
                if items[i].price > items[j].price:
                    items.insert(j, items[i])
                    del items[i+1]

        for i in range(len(seeds)):
            self._graph.nodes[seeds[i]]["desired_set"] = items[i]
            
    def _propagate(self, src, det, itemset):
        '''
            若影響成功則把itemset放到det節點的desired_set, 並且將edge的is_tested設定為true後, 回傳影響結果
        '''
        edge = self._graph.edges[src, det]
        if not edge["is_tested"]:
            edge["is_tested"] = True
            if self._graph.convertDirected(): # 如果原圖是無向圖, 則此條邊的另一個方向也無法再使用
                self._graph.edges[det, src]["is_tested"] = True

            # the attribute "is_active" is for tracing case easily 
            condition = edge["is_active"] if "is_active" in edge else uniform(0, 1) <= edge["weight"]
            if condition:
                self._graph.nodes[det]["desired_set"] = self._itemset.union(
                    self._graph.nodes[det]["desired_set"],
                    itemset
                )
                return True
                
        return False


    def diffusion(self, tagger=None):
        '''
            NOTE: 第一個seed會在買完後就傳遞影響力, 假如被傳遞的節點為第二個seed, 那後來seed的desired set
            不只會包含他被分配到的商品, 也會包含seed影響成功的商品。似乎在t step所有被影響成功的節點買完後,
            才開始傳遞影響力比較合理? 
        '''
        if not self._seeds:
            k = min(self._itemset.size, self._graph.number_of_nodes())

            # list of the seeds is sorted by out-degree.
            self.selectSeeds(k)
        
        if self._graph.nodes[self._seeds[0]]["desired_set"] == None:
            # List of single items.
            items = [self._itemset[id] for id in list(self._itemset.PRICE.keys())]

            self.allocate(self._seeds, items)

        # push the node who will adopt items at this step
        adoptionQueue = Queue()
        self.influencedNodes = list()
        for seed in self._seeds:
            adoptionQueue.put((None, seed, 1))
            self.influencedNodes.append(seed)
       
        # push the node who have adopted items at this step
        propagatedQueue = Queue()

        # Loop until no one adopted items at the previous step
        step = 0
        while not adoptionQueue.empty():
            
            # Loop until everyone check to decide whether adopt items
            while not adoptionQueue.empty():
                src, det, path_prob = adoptionQueue.get()
                node_id = det
                
                trade = self._user_proxy.adopt(node_id)
                
                # 如果沒購買任何東西則跳過此使用者不做後續的流程
                if trade == None:
                    if "TagNonActive" in tagger:
                        tagger["TagNonActive"].tag(node=self._graph.nodes[node_id])
                    continue

                '''
                    taggerParam = {
                        mainItemset: 主商品組合
                        seed: 節點屬於哪個種子群 (Algorithm) 
                        expectedProbability: 影響期望值
                        coupon: 該節點使用哪個優惠方案
                        node: 節點
                    }
                '''
                
                # logging.debug("Parameters of tagger------------------------------ ")
                # for k, v in trade.items():
                #     logging.debug("{0}: {1}".format(k, v))
                # logging.debug("--------------------------------")

                trade["src"] = src
                trade["det"] = node_id
                
                if tagger != None:
                    tagger.tag(trade, node_id=node_id, node=self._graph.nodes[node_id], path_prob=path_prob)

                propagatedQueue.put((node_id, trade["decision_items"], path_prob))
                adoptionQueue.task_done()

            step += 1
            while not propagatedQueue.empty():
                node_id, tradeOff_items, path_prob = propagatedQueue.get()
                for out_neighbor in self._graph.neighbors(node_id):
                    is_activated  = self._propagate(node_id, out_neighbor, tradeOff_items)
                    if is_activated:
                        weight = self._graph.edges[node_id, out_neighbor]["weight"]
                        adoptionQueue.put((node_id, out_neighbor, path_prob*weight))
                        self.influencedNodes.append(out_neighbor)
                propagatedQueue.task_done()
    
    def DeterministicDiffusion(self, depth:int, tagger=None):

        if not self._seeds:
            k = min(self._itemset.size, self._graph.number_of_nodes())

            # list of the seeds is sorted by out-degree.
            self.selectSeeds(k)
        
        if self._graph.nodes[self._seeds[0]]["desired_set"] == None:
            # List of single items.
            items = [self._itemset[id] for id in list(self._itemset.PRICE.keys())]

            self.allocate(self._seeds, items)

        # push the node who will adopt items at this step
        adoptionQueue = Queue()
        for seed in self._seeds:
            adoptionQueue.put((None, seed))
        
        # push the node who have adopted items at this step
        propagatedQueue = Queue()

        # Loop until no one adopted items at the previous step
        step = 0
        while not adoptionQueue.empty() and step <= depth:
            
            # Loop until everyone check to decide whether adopt items
            while not adoptionQueue.empty():
                src, det = adoptionQueue.get()
                node_id = det
                
                trade = self._user_proxy.adopt(node_id)
                
                # 如果沒購買任何東西則跳過此使用者不做後續的流程
                if trade == None:
                    if "TagNonActive" in tagger:
                        tagger["TagNonActive"].tag(node=self._graph.nodes[node_id])
                    continue

                '''
                    taggerParam = {
                        mainItemset: 主商品組合
                        seed: 節點屬於哪個種子群 (Algorithm) 
                        expectedProbability: 影響期望值
                        coupon: 該節點使用哪個優惠方案
                        node: 節點
                    }
                '''

                trade["src"] = src
                trade["det"] = node_id
                

                if tagger != None:
                     tagger.tag(trade, node_id=node_id, node=self._graph.nodes[node_id])

                propagatedQueue.put((node_id, trade["decision_items"]))
                adoptionQueue.task_done()

            step += 1
            while not propagatedQueue.empty():
                node_id, decision_items = propagatedQueue.get()
                for out_neighbor in self._graph.neighbors(node_id):
                    is_activated  = self._propagate(node_id, out_neighbor, decision_items)

                    self._graph.edges[node_id, out_neighbor]["is_tested"] = False
                    if self._graph.convertDirected():
                        self._graph.edges[det, src]["is_tested"] = False

                    if is_activated:
                        adoptionQueue.put((node_id, out_neighbor))
                propagatedQueue.task_done()
    # def save(self, dir_path):

    #     filename = dir_path + self.name
        
    #     def save_graph(G, filename):
    #         def  stringizer(value):
    #             if isinstance(value, (Itemset, Coupon)):
    #                 return str(value)
    #             elif value == None:
    #                 return ""

    #             return value
            
    #         write_gml(G, filename + ".graph", stringizer)

    #     def save_items(itemset: ItemsetFlyweight, filename):
    #         '''
    #             The first column is the asin of items, then price and the others are topics.
    #         '''
    #         with open(filename + ".items", 'w', encoding="utf8", newline="") as f:
    #             f.write("number {0}\n".format(len(itemset.PRICE)))
    #             f.write("asin,price,topic1,topic2,...\n")
    #             asinList = list(itemset.PRICE.keys())
    #             for asin in asinList:
    #                 f.write(asin + "," + str(itemset.PRICE[asin]) + ",")
    #                 for topic in itemset.TOPIC[asin]:
    #                     f.write(str(topic) + ",")
    #                 f.write("\n")

    #             save the relation of all items
    #             for x in asinList:
    #                 for y in asinList:
    #                     f.write(str(itemset._relation[x][y]))
    #                     f.write(" ")
    #                 f.write("\n")

    #     def save_coupons(coupons, filename):
    #         with open(filename + ".coupons", "w", encoding="utf8") as f:
    #             for coupon in coupons:
    #                 f.write(str(coupon))
        
    #     save_graph(self._graph, filename)
    #     save_items(self._itemset, filename)
    #     save_coupons(self._coupons, filename)
        
    # @staticmethod
    # def load(modelname, path):
        
    #     def load_graph(filename):
    #         sn_graph = SN_Graph()
    #         graph = read_gml(filename + ".graph")

    #         for src, det, data in graph.edges(data=True):
    #             sn_graph.add_edge(src, det, **data)
            
    #         for node, data in graph.nodes(data=True):
    #             if node not in sn_graph:
    #                 sn_graph.add_node(node, **data)
    #             else:
    #                 set_node_attributes(sn_graph, {node:data})
                    
    #         return sn_graph

    #     def load_items(filename):
                    
    #         prices = dict()
    #         topics = dict()
    #         relation = dict()
    #         asinList = list()
    #         with open(filename + ".items", "r") as f:
    #             number = f.readline().split(" ")[1]
    #             header = f.readline()
    #             for i in range(int(number)):
    #                 line = f.readline()
    #                 asin, price, *topic = line.split(",")
    #                 asinList.append(asin)
    #                 prices[asin] = float(price)
    #                 topics[asin] = [float(t) for t in topic[:-1]] # exclude new line

    #             for x in asinList:
    #                 if x not in relation:
    #                     relation[x] = dict()

    #                 line = f.readline().split(" ")
    #                 for j in range(int(number)):
    #                     y = asinList[j]
    #                     relation[x][y] = float(line[j])

    #         return ItemsetFlyweight(prices, topics, pd.DataFrame.from_dict(relation))

    #     def load_coupons(filename):
    #         coupons = []
    #         with open(filename + ".coupons", "r") as f:
    #             for line in f:
    #                 attr = line.split(",")
    #                 coupons.append(Coupon(
    #                                 float(attr[0]),
    #                                 attr[1], 
    #                                 float(attr[2]),
    #                                 attr[3])
    #                             )
            
    #         return coupons

    #     filename = path + modelname
    #     graph = load_graph(filename)
    #     itemset = load_items(filename)
    #     coupons = load_coupons(filename)

    #     for node, data in graph.nodes(data=True):
    #         for key, value in data.items():
    #             if key == "desired_set" or key == "adopted_set":
    #                 graph.nodes[node][key] = itemset[value] if value != None else None
    #             elif key == "adopted_records":
    #                 for i in range(0, len(value), 3):
    #                     graph.nodes[node][key][i] = itemset[value[i]]

    #                     c = None
    #                     if value[i+1] != "":
    #                         coupon_args = value[i+1].split(",")
    #                         c = Coupon(
    #                                     float(coupon_args[0]),
    #                                     itemset[coupon_args[1]],
    #                                     float(coupon_args[2]),
    #                                     itemset[coupon_args[3]],
    #                                     )

    #                     graph.nodes[node][key][i+1] = c
    #                     graph.nodes[node][key][i+2] = float(value[i+2])

    #     return DiffusionModel(modelname, graph, itemset, coupons)
