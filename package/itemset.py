from networkx.classes.function import number_of_nodes
import numpy as np
import pandas as pd
import itertools
import json
from random import random
from itertools import combinations
import math

from package.topic import TopicModel

class Itemset():
  
    '''
        Structure of Itemset
    '''

    def __init__(self, numbering, price, topic):
        self.numbering = set(numbering)  if numbering else set()
        self.price = price
        self.topic = topic

    def __eq__(self, other):
    
        '''
          Args:
            other (Itmeset,set)
        '''
        num_set = set()
    
        if isinstance(other, Itemset):
            num_set = other.numbering
        elif isinstance(other, set):
            num_set = other
        elif other != None:
            raise TypeError("Both of variables should be \"Itemset\" class or set type")

        return self.numbering == num_set
    
    def __len__(self):
        return len(self.numbering)

    def __str__(self):
        sortedNum = sorted(self.numbering)
        return " ".join(str(num) for num in sortedNum)
    
    def __hash__(self) -> int:
        return hash(self.__str__)
    
    def empty(self):
        return len(self.numbering) == 0

class ItemRelation():
    def __init__(self, relation: pd.DataFrame = None):
        self._relation = relation

        if relation is not None:
            self._normalize()

    def __str__(self) -> str:
        sorted_index = self._relation.sort_index(ascending=True)
        sorted_index = sorted_index.reindex(sorted(sorted_index.columns), axis=1)

        return str(sorted_index)

    def __getitem__(self, key):
        x, y = key

        if x not in self._relation:
            raise KeyError("{0} is not in the realtion matrix.")
        if y != None:
            if y not in self._relation[x]:
                return self._transform(0)
            return self._relation[x][y]
        
        return self._relation[x]
    
    def __iter__(self):
        for key in self._relation:
            yield self._relation[key]

    def construct(self, dataset):
        '''
            計算產品替代互補關係程度矩陣。對於任意倆倆商品x,y如下計算
            1. 買了x也買了y, 則R_x,y + 1; 看了x卻買了y, 則R_x,y - 1
            2. 做 max-min 正規化
            3. 對於每一個元素t做 1 + sigmoid(t)
        '''

        def counting(_relation, src_asin, collection, buyOrView):
            op = 1 if buyOrView == "buy" else -1

            for dest_asin in collection:
                if dest_asin not in _relation[src_asin]:
                    _relation[src_asin][dest_asin] = op
                else:
                    _relation[src_asin][dest_asin] += op 

        self._relation = dict()
        with open(dataset, "r", encoding="utf-8") as f:
            for line in f:
                asin, also_view, also_buy, *other = line.split(",")
                if asin not in self._relation:
                    self._relation[asin] = dict()

                also_view_set = also_view.split(" ")
                also_buy_set = also_buy.split(" ")
                counting(self._relation, asin, also_buy_set, "buy")
                counting(self._relation, asin, also_view_set, "view")
        
        self._relation = pd.DataFrame.from_dict(self._relation)
        self._normalize()
    
    def _transform(self, param: int|float|pd.DataFrame):
        return 1 + (1/ (1+np.exp(-param)))
    
    def _normalize(self):
        def min_max(frame):
            maxValue = frame.abs().max().max()
            
            return frame / maxValue
            
        self._relation = min_max(self._relation.fillna(0))
        self._relation = self._transform(self._relation)
        
        for numbering, value in self._relation.items():
            # if x.numbering == y.numbering, assign nan
            value[numbering] = math.nan

class ItemsetFlyweight():
  
    '''
        For creating itemset instance with flyweight pattern 
    '''
    
    def __init__(self, prices:dict, topic:TopicModel|dict, relation:ItemRelation = None) -> None:
        '''
            If item_file is None, prices and topics should be set.
            Args:
                price (dict): mapping from id to price
                topic (TopicModel)
                relation (dict|pd.DataFrame): It is a two dimensional matrix which the indices are asin of the item.
        '''
        self.PRICE = prices # dict
        self.TOPIC = topic if type(topic) == dict else topic.getItemsTopic()
        self._relation = relation
        self.size = len(list(prices.values()))
        self._map = dict()

        for key in self.TOPIC.keys():
            # initialize
            ids = key.split(" ")
            self.__getitem__(ids)
                
    def __getitem__(self, ids) -> Itemset:
        '''
            Args:
                ids(str, array-like): all of ids in the itemset 
        '''
        if type(ids) == Itemset:
            return ids
            
        sortedNum = ids.split(" ") if type(ids) == str else list(ids)

        sortedNum = sorted(sortedNum)
        key = " ".join(str(num) for num in sortedNum)
        itemset = None

        if key in self._map:
            itemset = self._map[key]

        else:
            itemset = Itemset(sortedNum,
                        sum([self.PRICE[i] for i in sortedNum]),
                        self._aggregateTopic(sortedNum) if key not in self.TOPIC else self.TOPIC[key] # the argument is useless because it's random
                        )
            self._map[key] = itemset

        return itemset
    
    def __iter__(self):
        numbering_of_items = list(self.PRICE.keys())

        for numbering, obj in self.powerSet(numbering_of_items):
            yield numbering, obj


    def _aggregateTopic(self, collection):
        aggregated = [0]*len(self._map[collection[0]].topic)
        coeff_list = []
        
        for i in collection:
            coeff_list.append(0)
            for j in collection:
                if j != i:
                    coeff_list[-1] += self._relation[j, i]

        normDenominator = sum(coeff_list)
        for i in range(len(collection)):
            topic = self._map[collection[i]].topic
            for j in range(len(topic)):
                aggregated[j] += ((topic[j]*coeff_list[i])/normDenominator)
                
        return aggregated

    @staticmethod
    def _toSet(a):

        a_set = set()
        if isinstance(a, Itemset):
            a_set = a.numbering
        elif a == None or len(a) == 0:
            a_set = set()
        else:
            a_set = set(a)
        return a_set

    def union(self, a, b):
        '''
            Union two itemset
            a(set, Itemset)
            b(set, Itemset)
        '''
    
        union_set = set()
        a_set = ItemsetFlyweight._toSet(a)
        b_set = ItemsetFlyweight._toSet(b)

        union_set = a_set.union(b_set)
        return self.__getitem__(union_set)

    def intersection(self, a, b):
        '''
            intersection two itemset
            a(set, Itemset, None)
            b(set, Itemset, None)
        '''

        a_set = ItemsetFlyweight._toSet(a)
        b_set = ItemsetFlyweight._toSet(b)
        intersection = b_set.intersection(a_set)

        return self.__getitem__(intersection) if len(intersection) != 0 else None

    def difference(self, a, b):

        a_set = ItemsetFlyweight._toSet(a)
        b_set = ItemsetFlyweight._toSet(b)
    
        minus = a_set.difference(b_set)
        return self.__getitem__(minus) if len(minus) != 0 else None
    
    def issubset(self, a, b) -> bool:
        '''
            Return:
                If this itemset is subeset of "other", return true otheriwse false.
        '''
        a_set = ItemsetFlyweight._toSet(a)
        b_set = ItemsetFlyweight._toSet(b)

        return a_set.issubset(b_set)

    def issuperset(self, a, b) -> bool:

        a_set = ItemsetFlyweight._toSet(a)
        b_set = ItemsetFlyweight._toSet(b)

        return a_set.issuperset(b_set)
    
    def powerSet(self, items) -> iter:
        '''
            Exclude empty itemset
        '''
        for size_itemset in range(len(items)):
            for combination in combinations(ItemsetFlyweight._toSet(items), size_itemset + 1):
                itemset = self.__getitem__(combination)
                yield str(itemset), itemset
        
    def maxTopicValue(self, userTopic):
        
        maxValue = 0
        maxObj = None

        for id, obj in self.__iter__():

            value = 0
            for t in range(len(userTopic)):
                value += obj.topic[t]*userTopic[t]

            #print("{0}: {1}".format(id, value))
            if value > maxValue:
                maxValue, maxObj = value, obj

        #print("max {0}: {1}".format(str(maxObj), maxValue))
        return maxObj

    def maxSupersetValue(self, userTopic, mainItemset):
        
        maxValue = 0
        maxObj = None

        for id, obj in self.__iter__():
            if self.issuperset(obj, mainItemset):
                value = 0
                for t in range(len(userTopic)):
                    value += obj.topic[t]*userTopic[t]

                #print("{0}: {1}".format(id, value))
                if value > maxValue:
                    maxValue, maxObj = value, obj

        #print("max {0}: {1}".format(str(maxObj), maxValue))
        return maxObj

    def sortByPrice(self, arr, reverse=False):
        return sorted(arr, key=lambda x: self.__getitem__(x).price, reverse=reverse)