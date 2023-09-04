import logging
import networkx as nx
import math
import numpy as np

from package.itemset import ItemsetFlyweight
from package.social_graph import SN_Graph

class Tagger:
    def __init__(self):
        self._next = None
        self._params = dict()
        self._map = dict()

    def setNext(self, tag):
        
        lastObj = self
        while lastObj._next is not None:
            lastObj = lastObj._next
        
        lastObj._next = tag
        self._map[type(tag).__name__] = tag

    def getNext(self):
        return self._next
    
    def setParams(self, param=dict(), **kwargs):
        for k, v in param.items():
            self._params[k] = v

        for k, v in kwargs.items():
            self._params[k] = v
        
    def tag(self, params, **kwargs):
        ''' implement in child class'''
        self.setParams(params, **kwargs)

        if self._next:
            self._next.tag(self._params)
    
    def __getitem__(self, key):
        return self._map[key]

class TagMainItemset(Tagger):
    def __init__(self):
        super().__init__()
        self.table = dict()

    def __getitem__(self, group):
        return self.table[group]

    def tag(self, params=dict(), **kwargs):
        self.setParams(params, **kwargs)

        itemset = self._params["mainItemset"]
        node_id = self._params["node_id"]

        seed = None
        for num, nodes in self._params["group"].items():
            if node_id in nodes:
                seed = num
                break

        expectedProbability = self._params["max_expected"][node_id]
        ids = str(itemset)

        if seed not in self.table:
            self.table[seed] = dict()
        
        if ids not in self.table[seed]:
            self.table[seed][ids] = expectedProbability
        else:
            self.table[seed][ids] += expectedProbability

        
        # Temporarily delete the key which is None
        for group in self.table.keys():
            if "None" in self.table[group]:
                del self.table[group]["None"]

        super().tag(self._params)

    def maxExcepted(self, group):
        if len(self.table[group]) == 0:
            return None
        return max(self.table[group], key=self.table[group].get)
    
class TagAppending(Tagger):
    '''
        Appending 與 Addtional 不同, Appeding是沒有優惠方案下與主商品搭配時, 效益最高的附加品。
    '''
    def __init__(self, itemset:ItemsetFlyweight):
        super().__init__()
        self.table = dict()
        self._itemset = itemset

    def __iter__(self):
        for seed, exceptedAppending in self.table.items():
            yield seed, exceptedAppending
    
    def __getitem__(self, group):
        return self.table[group]
    
    def tag(self, params=dict(), **kwargs):

        self.setParams(params, **kwargs)
        
        node_id = self._params["node_id"]
        mainItemset = self._params["mainItemset"]

        seed = None
        for num, nodes in self._params["group"].items():
            if node_id in nodes:
                seed = num
            
        expectedProbability = self._params["max_expected"][node_id]

        if seed not in self.table:
            self.table[seed] = dict()
        
        obj = self._itemset.maxSupersetValue(params["node"]["topic"], mainItemset)
        logging.debug("Node: {0}, Maxum benfit itemset: {1}, Probability {2}".format(node_id, str(obj), expectedProbability))

        addItemset = self._itemset.difference(obj, mainItemset)
        ids = str(addItemset)

        if ids not in self.table[seed]:
            self.table[seed][ids] = expectedProbability
        else:
            self.table[seed][ids] += expectedProbability

        # Temporarily delete the key which is None
        for group in self.table.keys():
            if "None" in self.table[group]:
                del self.table[group]["None"]
                
        super().tag(self._params)
    
    def maxExcepted(self, group):
        if len(self.table[group]) == 0:
            return None
        
        return max(self.table[group], key=self.table[group].get)

class TagRevenue(Tagger):
    def __init__(self, graph, seeds, max_expected=dict()):
        super().__init__()
        self._amount = 0
        self._expected_amount = 0
        self._seeds = seeds
        self._compile_graph, self._max_expected = SN_Graph.compile_max_product_graph(graph, self._seeds)
        self._graph = graph
        for seed in self._seeds:
            self._max_expected[seed] = 1

    def tag(self, params, **kwargs):

        self.setParams(params, **kwargs)
        
        det = self._params["det"]
        # price multi maximum expected probability
        self._expected_amount += params["amount"]*self._max_expected[det]
        self._amount += params["amount"]
        if "max_expected" not in self._params:
            self._params["max_expected"] = self._max_expected
        super().tag(self._params)

    def amount(self):
        return self._amount
    
    def expected_amount(self):
        return self._expected_amount
    
    def avg(self, times):
        self._amount = self._amount/times
        self._expected_amount = self._expected_amount/times
    
class TagActiveNode(Tagger):
    def __init__(self):
        super().__init__()
        self._amount = 0
        self._expected_amount = 0
        self._distirbution = [0 for i in np.arange(0, 1.1, 0.1, dtype=float)]

    def tag(self, params, **kwargs):
        self.setParams(params, **kwargs)
        if "max_expected" not in self._params:
            raise ValueError("You should calculate maximum expected probability before counting active node.")
                             
        node = self._params["node"]
        node_id = self._params["node_id"]
        if len(node["adopted_records"]) == 1:
            self._expected_amount += self._params["max_expected"][node_id]
            self._distirbution[math.floor(self._params["max_expected"][node_id]*10)] += 1
            self._amount += 1
        super().tag(self._params)

    def amount(self):
        return self._amount
    
    def expected_amount(self):
        return self._expected_amount
    
    def avg(self, times):
        self._amount = self._amount/times
        self._expected_amount = self._expected_amount/times
    
    def distribution(self): # pragma: no cover
        return self._distirbution
