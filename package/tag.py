from itemset import ItemsetFlyweight
import logging
import networkx as nx

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
        seed = self._params["belonging"][node_id]
        expectedProbability = self._params["expectedProbability"][seed][node_id]
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
        seed = self._params["belonging"][node_id]
        expectedProbability = self._params["expectedProbability"][seed][node_id]

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
    def __init__(self, graph, shortest_path_len):
        super().__init__()
        self._counting = 0
        self._graph = graph
        self._shortest_path_len = shortest_path_len

    def tag(self, params, **kwargs):

        self.setParams(params, **kwargs)

        src, det = self._params["src"], self._params["det"]
        # if src is None, it is a seed
        if src != None:
            if src not in self._shortest_path_len:
                self._shortest_path_len[src] = dict()
    
            if det not in self._shortest_path_len[src]:
                self._shortest_path_len[src][det] = self._graph.caculate_shortest_path_length(src, det)

        expectedProbability = self._shortest_path_len[src][det] if src != None else 1
        self._counting += params["amount"]*expectedProbability
        super().tag(self._params)

    def amount(self):
        return self._counting

class TagActivatedNode(Tagger):
    def __init__(self):
        super().__init__()
        self._count = 0

    def tag(self, params, **kwargs):
        self.setParams(params, **kwargs)
        
        if len(self._params["node"]["adopted_records"]) == 1:
            self._count += 1

        super().tag(self._params)

    def amount(self):
        return self._count