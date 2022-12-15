from math import degrees
from queue import Queue
from random import uniform, random
import logging
import re
from networkx import NetworkXError, write_gml, read_gml
from os.path import exists

# custom package
from user_proxy import UsersProxy
from itemset import ItemsetFlyweight, Itemset
from coupon import Coupon
from social_graph import SN_Graph

class DiffusionModel():
    def __init__(self, name, graph:SN_Graph, items, coupons) -> None:
        self._graph = graph
        self.name = name

        self._itemset = ItemsetFlyweight(items["price"], items["topic"])
        for node in self._graph:
            self._graph.nodes[node]['topic'] = self._randomTopic(len(items["topic"]['0']))

        for coupon in coupons:
            coupon.accItemset = self._itemset[coupon.accItemset]
            coupon.disItemset = self._itemset[coupon.disItemset]

        self._coupons = coupons
        self._user_proxy = UsersProxy(self._graph, self._itemset, self._coupons)
    
    
    def _randomTopic(self, T:int):
        topic = [random() for i in range(T)]
        return [topic[i]/sum(topic) for i in range(T)]

    def _selectSeeds(self, k:int) -> list:
        '''
            Return:
                list: an ordered list of nodes which is sorted by degrees
        '''
        return self._graph.top_k_nodes(k)

    def _allocate(self, seeds, items):
        '''
            Args:
                seeds (list of tuple): A list of seeds with degree
                items (list of Itemset)
            將商品分配給種子節點，並且放進對應的 desired_set。
            分配策略: 單價越高的商品分給degree越高的節點
        '''
        sortedItems = []
        for item in items:
            insertPos = 0
            for i in range(len(sortedItems)):
                if item.price > sortedItems[i].price:
                    insertPos = i
            sortedItems.insert(insertPos, item)

        for i in range(len(seeds)):
            self._graph.nodes[seeds[i][0]]["desired_set"] = sortedItems[i]
    
    def _propagate(self, src, det, itemset):
        '''
            若影響成功則把itemset放到det節點的desired_set，並且將edge的is_tested設定為true後，回傳影響結果
        '''
        edge = self._graph.edges[src, det]
        if not edge["is_tested"]:
            self._graph.edges[src, det]["is_tested"] = True
            if uniform(0, 1) <= edge["weight"]:
                self._graph.nodes[det]["desired_set"] = self._itemset.union(
                    self._graph.nodes[det]["desired_set"],
                    itemset
                )
                return True
                
        return False

    def diffusion(self):

        k = min(self._itemset.number, self._graph.number_of_nodes())

        # List of single items.
        items = [self._itemset[i] for i in range(k)]

        # list of the seeds is sorted by out-degree.
        seeds = self._selectSeeds(k) 
        self._allocate(seeds, items)
        for seed, degree in seeds:
            logging.debug("Allocate {0} to {1}".format(self._graph.nodes[seed]["desired_set"], seed))
        logging.info("Allocation is complete.")

        propagatedQueue = Queue()
        for seed, out_degree in seeds:
            propagatedQueue.put(seed)
        
        while not propagatedQueue.empty():
            node_id = propagatedQueue.get()
            
            trade = self._user_proxy.adopt(node_id)
            logging.info("user: {0}, traded items:{1}".format(node_id, trade["decision_items"]) )

            # 如果沒購買任何東西則跳過此使用者不做後續的流程
            if trade == None:
                continue
            
            is_activated = False
            for out_neighbor in self._graph.neighbors(node_id):
                is_activated  = self._propagate(node_id, out_neighbor, trade["decision_items"])
                logging.info("{0} tries to activate {1}: {2}".format(node_id, out_neighbor, is_activated))
                if is_activated:
                    logging.debug("{0}'s desired_set: {1}".format(out_neighbor, self._graph.nodes[out_neighbor]["desired_set"]))
                    propagatedQueue.put(out_neighbor)
    
    def save(self, dir_path):

        filename = dir_path + self.name
        
        def save_graph(G, filename):
            def  stringizer(value):
                if isinstance(value, Itemset):
                    return str(value)
                elif value == None:
                    return ""
                
                return value
            
            write_gml(G, filename + ".graph", stringizer)

        def save_items(itemset: ItemsetFlyweight, filename):
            '''
                The first column is the price of items, and the others are topics.
            '''
            with open(filename + ".items", 'w', encoding="utf8", newline="") as f:
                for i in range(itemset.numbering):
                    f.write(str(itemset.PRICE[i]) + ",")
                    for topic in itemset.TOPIC[str(i)]:
                        f.write(str(topic) + ",")
                    f.write("\n")

        def save_coupons(coupons, filename):
            with open(filename + ".coupons", "w", encoding="utf8") as f:
                for coupon in coupons:
                    f.write(str(coupon))
        
        save_graph(self._graph, filename)
        save_items(self._itemset, filename)
        save_coupons(self._coupons, filename)

    def load(self, dir_path):

        filename = dir_path + self.name

        def load_graph(filename):
            graph = read_gml(filename + ".graph", destringizer=int)
            for node_id in graph:
                itemset_id = self._graph.nodes[node_id]["desired_set"]
                self._graph.nodes[node_id]["desired_set"] = self._itemset[itemset_id] if itemset_id != "" else None

                itemset_id = self._graph.nodes[node_id]["adopted_set"]
                self._graph.nodes[node_id]["adopted_set"] = self._itemset[itemset_id] if itemset_id != "" else None
            return graph

        def load_items(filename):
                    
            prices = []
            topics = []
            with open(filename + ".items", "r") as f:
                for line in f:
                    price, *topic = line.split(",")
                    prices.append(price)
                    topics.append(topic)

            return ItemsetFlyweight(prices, topics)

        def load_coupons(itemset: ItemsetFlyweight, filename):
            coupons = []
            with open(filename + ".coupons", "r") as f:
                for line in f:
                    attr = line.split(",")
                    coupons.append(Coupon(
                                    int(attr[0]),
                                    itemset[attr[1]], 
                                    int(attr[2]),
                                    itemset[attr[3]])
                                )
            
            return coupons

        self._graph = load_graph(filename)
        self._itemset = load_items(filename)
        self._coupons = load_coupons(self._itemset, filename)