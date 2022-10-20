from networkx.classes.function import number_of_nodes
import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy as np
import itertools
import json
from faker import Faker
from faker.providers import BaseProvider
from random import random
from itertools import combinations

class Itemset():
  
    '''
        Structure of Itemset
    '''

    def __init__(self, numbering):
        self.numbering = set(numbering)
        self.price = int
        self.topic = list

    def __eq__(self, other):
    
        '''
          Args:
            other (Itmeset,set,None): None is as empty itemset
        '''
        num_set = set()
    
        if isinstance(other, Itemset):
            num_set = other.numbering
        elif isinstance(other, set):
            num_set = other
        elif other != None:
            raise TypeError("Both of variables should be \"Itemset\" class or set type")

        return self.numbering == num_set

    def __str__(self):
        sortedNum = sorted(self.numbering)
        return " ".join(str(num) for num in sortedNum)

    def issubset(self, other) -> bool:
        '''
          Return:
              If this itemset is subeset of "other", return true otheriwse false.
        '''
        other_set = set
        other_set = other.numbering if isinstance(other, Itemset) else other

        return self.numbering.issubset(other_set)

    def issuperset(self, other) -> bool:

        other_set = set
        other_set = other.numbering if isinstance(other, Itemset) else other

        return self.numbering.issuperset(other_set)
    
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

    def __init__(self, number_of_items, price, number_of_topic, topic) -> None:
        '''
            if dataset is None, then randomly generate data
        '''
        self.PRICE = price
        self.TOPIC = topic
        self._map = {}

    def __getitem__(self, ids) -> Itemset:
        '''
            Args:
                ids(str, set, array-like or Itemset): all of ids in the itemset 
        '''
        sortedNum = []
        if type(ids) == str:
            sortedNum = [int(i) for i in ids.split(" ")]
        else: 
            sortedNum = list(ids)

        sortedNum = sorted(sortedNum)
        key = " ".join(str(num) for num in sortedNum)
        itemset = None

        if key in self._map:
            itemset = self._map[key]

        elif sortedNum != None and len(sortedNum) != 0:

            itemset = Itemset(sortedNum)
            itemset.price = sum([self.PRICE[i] for i in sortedNum])

          # the argument is useless because it's random
            itemset.topic = self._aggregateTopic(0,1) if key not in self.TOPIC else self.TOPIC[key]
            self._map[key] = itemset

        return itemset
  
    def items(self):

        number_of_items = len(self.PRICE)
        for size_itemset in range(number_of_items):
            for combination in combinations(range(number_of_items), size_itemset + 1):
                itemset = self.__getitem__(set(combination))
                yield str(itemset), itemset


    def _aggregateTopic(self, a, b):
        test = True
        if test:
            Z = len(self.TOPIC['0'])
            topic = [random() for i in range(Z)]
            denominator = sum(topic)
            return [t/denominator for t in topic]

  
    def union(self, a, b):
        '''
            Union two itemset
            a(set, Itemset)
            b(set, Itemset)
        '''
    
        union_set = set()

        a_set = a.numbering if isinstance(a, Itemset) else a
        b_set = b.numbering if isinstance(b, Itemset) else b


        union_set = a_set.union(b_set)
        return self.__getitem__(union_set)

    def intersection(self, a, b):
        '''
            intersection two itemset
            a(set, Itemset)
            b(set, Itemset)
        '''

        a_set = a.numbering if isinstance(a, Itemset) else a
        b_set = b.numbering if isinstance(b, Itemset) else b

        intersection = b_set.intersection(a_set)

        return self.__getitem__(intersection) if len(intersection) != 0 else None

    def difference(self, a, b):
        a_set = a.numbering if isinstance(a, Itemset) else a
        b_set = b.numbering if isinstance(b, Itemset) else b
    
        if b_set != None:
            minus = a_set.difference(b_set)
            return self.__getitem__(minus) if len(minus) != 0 else None
        else:
            return self.__getitem__(a_set)