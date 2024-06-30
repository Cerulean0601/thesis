from networkx.classes.function import number_of_nodes
import numpy as np
import pandas as pd
import itertools
import json
from random import random
from itertools import combinations
import math
from collections.abc import Iterable

from package.topic import TopicModel

class Itemset():
  
    '''
        Structure of Itemset
    '''

    def __init__(self, numbering:Iterable, price, topic, item2vec):
        if not isinstance(numbering, Iterable):
            raise TypeError('Numbering must be iterable')
        self.numbering = set(numbering)  if numbering else set()
        self.price = price
        self.topic = topic
        self.item2vec = item2vec
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
    
    def __hash__(self) -> int: # pragma: no cover
        return hash(self.__str__)

    def empty(self):
        return len(self.numbering) == 0

class ItemRelation():
    def __init__(self, relation: pd.DataFrame = None):
        self._relation = relation if relation else dict()
        self._data = dict()

    def __str__(self) -> str:
        sorted_index = self._relation.sort_index(ascending=True)
        sorted_index = sorted_index.reindex(sorted(sorted_index.columns), axis=1)

        return str(sorted_index)

    def __getitem__(self, key):
        x, y = key

        # if x not in self._relation:
        #     raise KeyError("{0} is not in the realtion matrix.")
        # if y != None:
        #     if y not in self._relation[x]:
        #         return self._transform(0)
        #     return self._relation[x][y]
        
        return self._relation.at[x, y]
    
    def __iter__(self):
        for key in self._relation:
            yield self._relation[key]
    
    def construct(self, dataset, substitute_coff=1, complementary_coff=1):
        
        with open(dataset, "r", encoding="utf-8") as f:
            for line in f:
                asin, also_view, also_buy, *other = line.split(",")
                if asin not in self._relation:
                    self._relation[asin] = dict()

                also_view_set = also_view.split(" ")
                for view_asin in also_view_set:
                    self._relation[asin][view_asin] = substitute_coff

                also_buy_set = also_buy.split(" ")
                for buy_asin in also_buy_set:
                    if buy_asin in self._relation[asin]:
                        self._relation[asin][buy_asin] = 1
                    else:
                        self._relation[asin][buy_asin] = complementary_coff

        self._relation = pd.DataFrame.from_dict(self._relation)   
        self._relation.fillna(1,inplace=True)

    # def construct(self, dataset):

    #     def counting(_relation, src_asin, collection, buyOrView):
    #         op = 1 if buyOrView == "buy" else -1

    #         for dest_asin in collection:
    #             if dest_asin not in _relation[src_asin]:
    #                 _relation[src_asin][dest_asin] = op
    #             else:
    #                 _relation[src_asin][dest_asin] += op 

    #     self._relation = dict()
    #     with open(dataset, "r", encoding="utf-8") as f:
    #         for line in f:
    #             asin, also_view, also_buy, *other = line.split(",")
    #             if asin not in self._relation:
    #                 self._relation[asin] = dict()

    #             also_view_set = also_view.split(" ")
    #             also_buy_set = also_buy.split(" ")
    #             self._data[asin] = dict()
    #             self._data["also_view"] = also_view_set
    #             self._data["also_buy"] = also_buy_set

    #             counting(self._relation, asin, also_buy_set, "buy")
    #             counting(self._relation, asin, also_view_set, "view")
        
    #    self._relation = pd.DataFrame.from_dict(self._relation)
    #    self._normalize()
    
    # def _transform(self, param: int|float|pd.DataFrame):
    #     return 1 + (1/ (1+np.exp(-param)))
    
    # def _normalize(self):
    #     def min_max(frame):
    #         maxValue = frame.abs().max().max()
            
    #         return frame / maxValue
            
    #     self._relation = min_max(self._relation.fillna(0))
    #     self._relation = self._transform(self._relation)
        
    #     for numbering, value in self._relation.items():
    #         # if x.numbering == y.numbering, assign nan
    #         value[numbering] = math.nan

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
        self._singleItems = sorted(list(self.PRICE.keys()))
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
            item2vec = []
            for singleItem in self.getSingleItems():
                if singleItem in sortedNum:
                    item2vec.append(1)
                else:
                    item2vec.append(0)
            itemset = Itemset(numbering=sortedNum,
                        price=sum([self.PRICE[i] for i in sortedNum]),
                        topic=self._aggregateTopic(sortedNum) if key not in self.TOPIC else self.TOPIC[key], # 如果組合只有一件商品，從TOPIC取就好
                        item2vec=item2vec
                        )
            self._map[key] = itemset

        return itemset
    
    def __iter__(self):
        numbering_of_items = list(self.PRICE.keys())

        for obj in self.powerSet(numbering_of_items):
            yield obj

    def getSingleItems(self) -> list[str]:
        return self._singleItems
    
    def _aggregateTopic(self, collection):
        if not collection:
            raise ValueError("The itemset which is aggreated is empty.\n")
        
        coeff = 0
        for i in collection:
            for j in collection:
                if j != i:
                    coeff += self._relation[i, j]

        # n 個商品會有n*(n-1)條關係
        n = len(collection)
        coeff = coeff / (n*(n-1))

        topic_list = []
        for asin in collection:
            topic_list.append(self._map[asin].topic)

        aggregated = [sum(x)*coeff/n for x in zip(*topic_list)]
        return aggregated
    
        # normDenominator = sum(coeff_list)
        # for i in range(len(collection)):
        #     topic = self._map[collection[i]].topic
        #     for j in range(len(topic)):
        #         aggregated[j] += ((topic[j]*coeff_list[i])/normDenominator)
                
        # return aggregated

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
    
        minus = a_set - b_set
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
                yield itemset
    
    # 計算兩個向量之間的Jaccard指數
    def jaccard_index(self, vec1, vec2) -> float:
        intersection = np.sum(np.logical_and(vec1, vec2))  # 計算交集
        union = np.sum(np.logical_or(vec1, vec2))         # 計算聯集
        return intersection / union

    # 找出具有最高Jaccard指數的向量對，如果有多個則判斷向量内積最高的選擇一對
    def find_max_jaccard_pair(self, collection):
        vectors = [itemset.item2vec for itemset in collection]
        max_jaccard = 0
        best_pair = (0,1)

        # 遍歷所有可能的向量對
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                jaccard = self.jaccard_index(vectors[i], vectors[j])
                if jaccard > max_jaccard:
                    max_jaccard = jaccard
                    best_pair = (i, j)
                elif jaccard == max_jaccard:
                    max_i, max_j = best_pair[0], best_pair[1]
                    dot = np.dot(collection[i].topic, collection[j].topic)
                    if dot > np.dot(collection[max_i].topic, collection[max_j].topic):
                        max_jaccard = jaccard
                        best_pair = (i, j)
    
        return best_pair, max_jaccard

    # 迭代地合併向量直到最高的Jaccard指數小於指定閾值，並記錄每個群組中的向量
    def merge_itemset_with_jaccard(self, collection, theta):
        groups = dict()
        for itemset in collection:  # 初始每個向量各自為一個群組
            groups[itemset] = [itemset] 

        while True:
            itemsets = list(groups.keys())
            if len(itemsets) <= 1 : # 全部的向量都被merge到同一個群組了
                break

            best_pair, max_jaccard = self.find_max_jaccard_pair(itemsets)
            if max_jaccard < theta:  # 如果最高的Jaccard指數小於閾值，停止合併
                break

            i, j = best_pair
            new_itemset = self.intersection(itemsets[i], itemsets[j])  # 合併選定的向量對

            # 更新群組信息
            new_group = groups[itemsets[i]] + groups[itemsets[j]]

            # 移除已合併的群組
            del groups[itemsets[i]]
            del groups[itemsets[j]]

            # 添加新合併的群組
            groups[new_itemset] = new_group

        return groups
        
    # def candidatedCouponItems(self, size, root):
    #     num_substitutions = size//2
    #     num_complementations = size - num_substitutions

    #     items = set()
    #     items.add(root)
    #     candidate_items = []

    #     while num_substitutions > 0:
    #         set_size = len(items)
    #         also_view = self._relation._data[root]["also_view"]
    #         items.update(also_view)
    #         num_substitutions -= (len(items) - set_size)
            
            
    # def maxTopicValue(self, userTopic):
        
    #     maxValue = 0
    #     maxObj = None

    #     for id, obj in self.__iter__():

    #         value = 0
    #         for t in range(len(userTopic)):
    #             value += obj.topic[t]*userTopic[t]

    #         #print("{0}: {1}".format(id, value))
    #         if value > maxValue:
    #             maxValue, maxObj = value, obj

    #     #print("max {0}: {1}".format(str(maxObj), maxValue))
    #     return maxObj

    # def maxSupersetValue(self, userTopic, mainItemset):
        
    #     maxValue = 0
    #     maxObj = None

    #     for id, obj in self.__iter__():
    #         if self.issuperset(obj, mainItemset):
    #             value = 0
    #             for t in range(len(userTopic)):
    #                 value += obj.topic[t]*userTopic[t]

    #             #print("{0}: {1}".format(id, value))
    #             if value > maxValue:
    #                 maxValue, maxObj = value, obj

    #     #print("max {0}: {1}".format(str(maxObj), maxValue))
    #     return maxObj

    def sortByPrice(self, arr, reverse=False):
        return sorted(arr, key=lambda x: self.__getitem__(x).price, reverse=reverse)