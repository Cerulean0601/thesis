from networkx.classes.function import number_of_nodes
import numpy as np
import pandas as pd
import itertools
import json
from faker.providers import BaseProvider
from random import random
from itertools import combinations
import math

from topic import TopicModel

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
    
    def empty(self):
        return len(self.numbering) == 0

class ItemProvider(BaseProvider):
    def __init__(self, k):
        self._k = k
      
    def prices(self, minPrice=1, maxPrice=1000) -> list:
        mid = int((minPrice + maxPrice)/2)
        p = [0]*(maxPrice - minPrice + 1)
        for i in range(mid):
            p[mid + i] = mid - i
            p[mid - i] = mid - i
        norm = sum(p)
        p = [weight/norm for weight in p]
        
        return np.random.choice(range(minPrice, maxPrice+1), size=self._k, p=p)
    '''
    def topicDistribution(self) -> list:
        return np.random.rand(1, self._k)
    '''

class ItemsetFlyweight():
  
    '''
        For creating itemset instance with flyweight pattern 
    '''
    
    def __init__(self, prices:dict, topic:TopicModel|dict) -> None:
        '''
            If item_file is None, prices and topics should be set.
            Args:
                price (dict): mapping from id to price
                topic (TopicModel)
        '''
        self.PRICE = prices # dict
        self.TOPIC = topic if type(topic) == dict else topic.getItemsTopic()
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
        sortedNum = [ids] if type(ids) == str else list(ids)

        sortedNum = sorted(sortedNum)
        key = " ".join(str(num) for num in sortedNum)
        itemset = None

        if key in self._map:
            itemset = self._map[key]

        else:
            itemset = Itemset(sortedNum,
                        sum([self.PRICE[i] for i in sortedNum]),
                        self._aggregateTopic(0,1) if key not in self.TOPIC else self.TOPIC[key] # the argument is useless because it's random
                        )
            self._map[key] = itemset

        return itemset
    
    def __iter__(self):
        numbering_of_items = list(self.PRICE.keys())

        for size_itemset in range(self.size):
            for combination in combinations(numbering_of_items, size_itemset + 1):
                itemset = self.__getitem__(combination)
                yield str(itemset), itemset


    def _aggregateTopic(self, a, b):
        '''
            Radomly generate the topic
        '''
        test = True
        if test:
            first_index = list(self.TOPIC.keys())[0]
            Z = len(self.TOPIC[first_index])
            topic = [random() for i in range(Z)]
            denominator = sum(topic)
            return [t/denominator for t in topic]

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

class ItemRelation():
    def __init__(self, relation = pd.DataFrame()):
        self._relation = relation
    
    def __str__(self) -> str:
        sorted_index = self._relation.sort_index(ascending=True)
        sorted_index = sorted_index.reindex(sorted(sorted_index.columns), axis=1)

        return str(sorted_index)

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

        def tranform(frame):
            return 1 + (1/ (1+np.exp(frame)) )
        
        def min_max(frame):
            maxValue = frame.max().max()
            minValue = frame.min().min()

            return (frame - maxValue) / (maxValue - minValue)

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
        # normalize
        self._relation = min_max(self._relation.fillna(0))
        print(self._relation)
        self._relation = tranform(self._relation)

        