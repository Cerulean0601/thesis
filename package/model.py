from math import degrees
from queue import Queue
from random import uniform, random
import logging
import re
from networkx import NetworkXError, write_gml, read_gml, set_node_attributes
from os.path import exists
from multiprocessing.pool import ThreadPool
import pandas as pd

# custom package
from user_proxy import UsersProxy
from itemset import ItemsetFlyweight, Itemset
from coupon import Coupon
from social_graph import SN_Graph
from topic import TopicModel
#from tag import Tagger, TagMainItemset, TagAppending

class DiffusionModel():
    def __init__(self, name, graph:SN_Graph, itemset, coupons=[]) -> None:
        self._graph = graph
        self.name = name

        self._itemset = ItemsetFlyweight(itemset["price"], itemset["topic"]) if type(itemset) == dict else itemset

        self._coupons = coupons
        self._user_proxy = UsersProxy(self._graph, self._itemset, self._coupons)
        self._seeds = []
        
    def getRevenue(self) -> list:

        def getEachNodeRevenue(node):
            records_len = len(node["adopted_records"])
            acc = 0
            for i in range(2, records_len, 3):
                acc += node["adopted_records"][i]
            
            return acc

        pool = ThreadPool()
        result = pool.map(getEachNodeRevenue, self._graph.nodes)
        pool.close()
        return result

    def getSeeds(self):
        return self._seeds

    def getGraph(self):
        return self._graph

    def setGraph(self, newGraph:SN_Graph):
        if not isinstance(newGraph, SN_Graph):
            raise TypeError("Replaced graph is not SN_Graph class.\n")
        
        self._user_proxy.setGraph(newGraph)
        self._graph = newGraph
        
    def getItemsetHandler(self):
        return self._itemset

    def getUserProxy(self):
        return self._user_proxy
    
    def setCoupons(self, coupons):
        self._coupons = coupons

    def _selectSeeds(self, k:int) -> list:
        '''
            Return:
                list: an ordered list of nodes which is sorted by degrees
        '''
        self._seeds = [seed[0] for seed in self._graph.top_k_nodes(k)]

    def allocate(self, seeds, items):
        '''
            Args:
                seeds (list): A list of seeds
                items (list of Itemset)
            將商品分配給種子節點，並且放進對應的 desired_set。
            分配策略: 單價越高的商品分給degree越高的節點
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
            self._graph.edges[src, det]["is_tested"] = True
            if self._graph.convertDirected(): # 如果原圖是無向圖, 則此條邊的另一個方向也無法再使用
                self._graph.edges[det, src]["is_tested"] = True

            if uniform(0, 1) <= edge["weight"]:
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
            self._selectSeeds(k)
        
        if self._graph.nodes[self._seeds[0]]["desired_set"] == None:
            # List of single items.
            items = [self._itemset[id] for id in list(self._itemset.PRICE.keys())]

            self.allocate(self._seeds, items)

        propagatedQueue = Queue()
        for seed in self._seeds:
            logging.debug("Allocate {0} to {1}".format(self._graph.nodes[seed]["desired_set"], seed))
            propagatedQueue.put(seed)

        logging.info("Allocation is complete.")
        
        while not propagatedQueue.empty():
            node_id = propagatedQueue.get()
            
            trade = self._user_proxy.adopt(node_id)
            
            # 如果沒購買任何東西則跳過此使用者不做後續的流程
            if trade == None:
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
            
            if trade["coupon"] is None or len(trade["coupon"]) == 0:
                trade["mainItemset"] = trade["decision_items"]
            logging.debug("Parameters of tagger------------------------------ ")
            for k, v in trade.items():
                logging.debug("{0}: {1}".format(k, v))
            logging.debug("--------------------------------")

            tagger.tag(trade, node_id=node_id, node=self._graph.nodes[node_id])
            logging.info("user {0} traded {1}".format(node_id, trade["tradeOff_items"]))
            
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
                if isinstance(value, (Itemset, Coupon)):
                    return str(value)
                elif value == None:
                    return ""

                return value
            
            write_gml(G, filename + ".graph", stringizer)

        def save_items(itemset: ItemsetFlyweight, filename):
            '''
                The first column is the asin of items, then price and the others are topics.
            '''
            with open(filename + ".items", 'w', encoding="utf8", newline="") as f:
                f.write("number {0}\n".format(len(itemset.PRICE)))
                f.write("asin,price,topic1,topic2,...\n")
                asinList = list(itemset.PRICE.keys())
                for asin in asinList:
                    f.write(asin + "," + str(itemset.PRICE[asin]) + ",")
                    for topic in itemset.TOPIC[asin]:
                        f.write(str(topic) + ",")
                    f.write("\n")

                # save the relation of all items
                for x in asinList:
                    for y in asinList:
                        f.write(str(itemset._relation[x][y]))
                        f.write(" ")
                    f.write("\n")

        def save_coupons(coupons, filename):
            with open(filename + ".coupons", "w", encoding="utf8") as f:
                for coupon in coupons:
                    f.write(str(coupon))
        
        save_graph(self._graph, filename)
        save_items(self._itemset, filename)
        save_coupons(self._coupons, filename)
        
    @staticmethod
    def load(modelname, path):
        
        def load_graph(filename):
            sn_graph = SN_Graph()
            graph = read_gml(filename + ".graph")

            for src, det, data in graph.edges(data=True):
                sn_graph.add_edge(src, det, **data)
            
            for node, data in graph.nodes(data=True):
                if node not in sn_graph:
                    sn_graph.add_node(node, **data)
                else:
                    set_node_attributes(sn_graph, {node:data})
                    
            return sn_graph

        def load_items(filename):
                    
            prices = dict()
            topics = dict()
            relation = dict()
            asinList = list()
            with open(filename + ".items", "r") as f:
                number = f.readline().split(" ")[1]
                header = f.readline()
                for i in range(int(number)):
                    line = f.readline()
                    asin, price, *topic = line.split(",")
                    asinList.append(asin)
                    prices[asin] = float(price)
                    topics[asin] = [float(t) for t in topic[:-1]] # exclude new line

                for x in asinList:
                    if x not in relation:
                        relation[x] = dict()

                    line = f.readline().split(" ")
                    for j in range(int(number)):
                        y = asinList[j]
                        relation[x][y] = float(line[j])

            return ItemsetFlyweight(prices, topics, pd.DataFrame.from_dict(relation))

        def load_coupons(filename):
            coupons = []
            with open(filename + ".coupons", "r") as f:
                for line in f:
                    attr = line.split(",")
                    coupons.append(Coupon(
                                    float(attr[0]),
                                    attr[1], 
                                    float(attr[2]),
                                    attr[3])
                                )
            
            return coupons

        filename = path + modelname
        graph = load_graph(filename)
        itemset = load_items(filename)
        coupons = load_coupons(filename)

        for node, data in graph.nodes(data=True):
            for key, value in data.items():
                if key == "desired_set" or key == "adopted_set":
                    graph.nodes[node][key] = itemset[value] if value != None else None
                elif key == "adopted_records":
                    for i in range(0, len(value), 3):
                        graph.nodes[node][key][i] = itemset[value[i]]

                        c = None
                        if value[i+1] != "":
                            coupon_args = value[i+1].split(",")
                            c = Coupon(
                                        float(coupon_args[0]),
                                        itemset[coupon_args[1]],
                                        float(coupon_args[2]),
                                        itemset[coupon_args[3]],
                                        )

                        graph.nodes[node][key][i+1] = c
                        graph.nodes[node][key][i+2] = float(value[i+2])

        return DiffusionModel(modelname, graph, itemset, coupons)
                        