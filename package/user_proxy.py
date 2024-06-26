import sys
from itertools import combinations
from multiprocessing.pool import Pool
from numpy import dot
from collections.abc import Iterator
from line_profiler import profile

from package.social_graph import SN_Graph
from package.cluster_graph import ClusterGraph
from package.itemset import ItemsetFlyweight, Itemset
from package.coupon import Coupon
class UsersProxy():
    '''
      使用者的購買行為，包含挑選主商品、額外購買、影響成功後的行為
    '''
    def __init__(self, graph: SN_Graph|ClusterGraph, itemset: ItemsetFlyweight, coupons, threshold = 0.0) -> None:
        self._graph = graph
        self._itemset = itemset
        self._coupons = coupons
        self._threshold = threshold

    def setGraph(self, newGraph:SN_Graph|ClusterGraph):
        self._graph = newGraph
    
    def getThreshold(self):
        return self._threshold
    
    def setCoupons(self, coupons:list[Coupon]):
        self._coupons = coupons

    def getCoupons(self) -> list[Coupon]:
        return self._coupons
    
    def _VP_ratio(self, user_id, itemset:str, mainItemset = None, coupon = None):
        if coupon == None:
            return self._mainItemsetVP(user_id, itemset)
        else:
            if mainItemset == None:
                raise ValueError("If there are any coupons, the mainItemset should be set")
            return self._addtionallyAdoptVP(user_id, mainItemset, itemset, coupon)
  
    def getVPsByUserId(self, user_id, coupon = None):
        mainItemset = None
        if coupon:
            mainItemset = self._adoptMainItemset(user_id)["items"]
        return {str(obj): self._VP_ratio(user_id=user_id, itemset=obj, mainItemset=mainItemset, coupon=coupon) for obj in self._itemset}
    
    def _similarity(self, user_id, itemset):

        if not isinstance(itemset, Itemset):
            itemset = self._itemset[itemset]

        itemset_topic = itemset.topic
        user_topic = self._graph.nodes[user_id]["topic"]

        sim = 0.0
        for i in range(len(itemset_topic)):
            sim += itemset_topic[i]*user_topic[i]

        return sim

    def _mainItemsetVP(self, user_id, itemset:str):
    
        '''
            挑選主商品時的計算公式

            Args:
                user_id (string): user id
                itemset (Itemset, set, array-like): which itemset for caculating

            Return:
                float: return the VP ratio with the pair in "main itemset" stage.
                If the result adopted itemset of the user minus the itemset is empty, return 0
        '''
        adopted = self._graph.nodes[user_id]["adopted_set"]
        if not isinstance(itemset, Itemset):
            itemset = self._itemset[itemset]

        result = self._itemset.difference(itemset ,adopted)
        if result:
            return self._similarity(user_id,itemset)/result.price
        else:
            return 0
  
    def _adoptMainItemset(self, user_id):
        '''
            使用者從 desired set 考量已購買過的商品以及CP值的商品組合, 選取CP值最高的商品組合當作主商品

            Args:
                user_id (str): user id
            Returns:
                (Itemset, float): If user's desired_set is empty, or the main itemset and adopted set is equivalance, 
                empty itemset will be returned. Otherwise, otherwise it will return the main itemset of this user with the value.
        '''

        if user_id not in self._graph:
            raise ValueError("The user id is not found.")
            

        desired_set = self._graph.nodes[user_id]["desired_set"]

        # items in the set had been adopted
        adopted_set = self._graph.nodes[user_id]["adopted_set"] 

        # If desired set is empty, or adopted set and desired set is equivalance, return None
        if desired_set is None or adopted_set == desired_set:
            return None

        if adopted_set is None:
            adopted_set = set()
            

        max_VP = sys.float_info.min
        maxVP_mainItemset = None

        for length in range(len(desired_set.numbering)):
            for X in combinations(desired_set.numbering, length+1):
                # the combination of iterator returns tuple type
                X = set(X)
                if self._itemset.issubset(adopted_set, X) and adopted_set != X:

                    VP = self._VP_ratio(user_id, X)
                    if VP > max_VP:
                        max_VP = VP
                        maxVP_mainItemset = X

        if maxVP_mainItemset is None or self._itemset[maxVP_mainItemset] == adopted_set:
            return None
        
        return {"items": self._itemset[maxVP_mainItemset], "VP": max_VP} if maxVP_mainItemset != None else None

    def _addtionallyAdoptVP(self, user_id:str, mainItemset:str|Itemset, itemset:str|Itemset, coupon):
        '''
            計算額外購買的VP值
            Args:
                user_id(str): node id
                mainItemset(Itemset,str): Main itemset of the node
                addtionalItemset(Itemset): 使用者在第二階段考量的商品組合
        '''
        mainItemset = self._itemset[mainItemset]
        itemset = self._itemset[itemset]
        
        mainItemset = self._itemset.difference(mainItemset, self._graph.nodes[user_id]["adopted_set"])
        accAmount = self._itemset.intersection(mainItemset, self._itemset[coupon.accItemset]) # 已累積的金額
        accAmount = 0 if accAmount == None else accAmount.price
        priceThreshold = coupon.accThreshold
        ratio = min(accAmount/priceThreshold, 1) # 滿額佔比 \phi

        # 扣除已擁有的商品後，實際交易的商品
        dealItemset = self._itemset.difference(itemset, self._graph.nodes[user_id]["adopted_set"])
        if dealItemset == None:
            raise ZeroDivisionError("Itemset and adopted set of the user should be equivalance.")

        amount = dealItemset.price
        sim = self._similarity(user_id, itemset)

        if dealItemset.price >= coupon.accThreshold:
            disItemset = self._itemset.intersection(dealItemset, coupon.disItemset)
            dealDiscount = min(disItemset.price, coupon.discount) if disItemset != None else 0
            amount -= dealDiscount

        return ratio*(sim/amount) if amount != 0 else sys.float_info.max
    
    @profile
    def _adoptAddtional(self, user_id, mainItemset):
        '''
            第二購買階段，即考量優惠方案的情況下

            Args:
                mainItemset(Itemset): 第一購買階段決定的商品組合，此組合為考量的商品加上曾購買過的商品
        '''
        
        def parallelAdopt(mainItemset, itemset_instance, coupon):
            result = None

            # X 必須超過滿額門檻
            diff_adopted = self._itemset.difference(itemset_instance, self._graph.nodes[user_id]["adopted_set"])
            intersection = self._itemset.intersection(diff_adopted, coupon.accItemset)
            
            if intersection != None and intersection.price >= coupon.accThreshold:
                VP = self._addtionallyAdoptVP(user_id, mainItemset, itemset_instance, coupon)
                result = {"items": itemset_instance, "VP": VP, "coupon": coupon}

            return result
        
        result = []
        for itemsetObj in self._itemset:
            # 商品組合必須是主商品的超集
            if self._itemset.issuperset(itemsetObj, mainItemset):
                for coupon in self._coupons:
                    #若主商品跟可累積商品的交集為空集合，則VP值等於0的情況下不需要考慮
                    if self._itemset.intersection(mainItemset, coupon.accItemset):
                        result.append(parallelAdopt(mainItemset, itemsetObj, coupon))
        
        maxVP = {"VP": sys.float_info.min}

        for comp_result in result:
            if comp_result != None and comp_result["VP"] > maxVP["VP"]:
                maxVP = comp_result

        return maxVP        
        # for key, itemset_instance in self._itemset:
        #     # Find the itemset X which is a superset of main itemset
        #     if self._itemset.issuperset(itemset_instance, mainItemset):
        #         for i in range(len(self._coupons)):
        #             # X 必須超過滿額門檻
        #             diff_adopted = self._itemset.difference(itemset_instance, self._graph.nodes[user_id]["adopted_set"])
        #             intersection = self._itemset.intersection(diff_adopted, self._coupons[i].accItemset)
                    
        #             if intersection != None and intersection.price > self._coupons[i].accThreshold:
        #                 VP = self._addtionallyAdoptVP(user_id, mainItemset, itemset_instance, self._coupons[i])

        #                 if VP > max_VP:
        #                     max_VP = VP
        #                     maxDict = {"items": itemset_instance, "VP": max_VP, "coupon": self._coupons[i]}

        #return maxDict
    
    def _discount(self, itemset, coupon):
        '''
            未滿足門檻會回傳0，否則回傳實際折扣金額。若可折扣金額大於可折扣商品的總價，
            則實際折扣金額等於可折扣商品的總價，反之等於可折扣金額。

            Args:
                itemset (Itemset): 必須排除已購買過的商品
            Return:
                int
        '''
        actuallyAcc = self._itemset.intersection(itemset, coupon.accItemset)

        if actuallyAcc == None or actuallyAcc.price < coupon.accThreshold:
            return 0

        actuallyDis = self._itemset.intersection(itemset, coupon.disItemset)
        return 0 if actuallyDis == None else min(actuallyDis.price, coupon.discount)
    def adopt(self, user_id):
        '''
            使用者購買行為, 若挑選的主商品皆為已購買過的商品則不會產生任何的購買行為, 並且回傳None。
            將實際交易的商品存到使用者的 adopted set, 回傳考量的商品組合, 實際交易的商品組合, 實際交易價格。

            Return:
                dict: 
                    decision_items 考量的商品組合
                    amount 交易金額，若有搭配優惠方案則已扣除折抵金額。

                None: 未發生購買行為
        ''' 
        mainItemset = self._adoptMainItemset(user_id)

        # 主商品皆為已購買過的商品
        # main itemset should be check whether is empty

        if mainItemset == None or self._itemset.issubset(mainItemset["items"], self._graph.nodes[user_id]["adopted_set"]):
            return None

        trade = dict()
        
        trade["mainItemset"] = mainItemset["items"]
        trade["decision_items"] = mainItemset["items"]
        trade["tradeOff_items"] = self._itemset.difference(trade["decision_items"], self._graph.nodes[user_id]["adopted_set"])
        trade["amount"] = trade["tradeOff_items"].price
        trade["coupon"] = None
        trade["VP"] = mainItemset["VP"]

        if self._coupons != None and len(self._coupons) != 0:
            addtional = self._adoptAddtional(user_id, mainItemset["items"])
            if trade["VP"] < addtional["VP"]:
                trade["VP"] = addtional["VP"]
                trade["decision_items"] = addtional["items"]
                trade["tradeOff_items"] = self._itemset.difference(trade["decision_items"], self._graph.nodes[user_id]["adopted_set"]) 
                trade["amount"] = trade["tradeOff_items"].price - self._discount(trade["tradeOff_items"], addtional["coupon"])
                trade["coupon"] = addtional["coupon"]

        if trade["VP"] >= self._threshold:
            self._graph.nodes[user_id]["adopted_set"] = self._itemset.union(
                                    self._graph.nodes[user_id]["adopted_set"],
                                    trade["decision_items"])

            self._graph.nodes[user_id]["adopted_records"].append([trade["tradeOff_items"], trade["coupon"], trade["amount"]])
            return trade
        else:
            return None
    
    def _min_discount(self, user_id, mainItemset, itemset) -> float:
        itemset = self._itemset[itemset]
        return itemset.price - (dot(itemset.topic, self._graph.nodes[user_id]["topic"])/self._VP_ratio(user_id, mainItemset))
    
    def discoutableItems(self, user_id, mainItemset: Itemset) -> Iterator[Itemset]:
        discoutable = []
        itemsetHandler = self._itemset
        node = self._graph.nodes[user_id]

        for itemset in itemsetHandler:
            if itemsetHandler.issuperset(itemset, mainItemset):
                if dot(mainItemset.topic, node["topic"]) < dot(itemset.topic, node["topic"]) or \
                self._VP_ratio(user_id, mainItemset) < self._VP_ratio(user_id, itemset):
                    yield itemset

