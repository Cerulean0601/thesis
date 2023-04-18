import sys
from itertools import combinations
import logging
from multiprocessing.pool import ThreadPool

from social_graph import SN_Graph
from itemset import ItemsetFlyweight, Itemset

class UsersProxy():
    '''
      使用者的購買行為，包含挑選主商品、額外購買、影響成功後的行為
    '''
    def __init__(self, graph: SN_Graph, itemset: ItemsetFlyweight, coupons, threshold = 0.0) -> None:
        self._graph = graph
        self._itemset = itemset
        self._coupons = coupons
        self._threshold = threshold

    def setGraph(self, newGraph:SN_Graph):
        if not isinstance(newGraph, SN_Graph):
            raise TypeError("Replaced graph is not SN_Graph class.\n")
        
        self._graph = newGraph
    
    def getThreshold(self):
        return self._threshold
    
    def replaceCoupons(self, coupons):
        self._coupons = coupons

    def _VP_ratio(self, user_id, itemset, mainItemset = None, coupon = None):
        if coupon == None:
            return self._mainItemsetVP(user_id, itemset)
        else:
            if mainItemset == None:
                raise ValueError("If there are any coupons, the mainItemset should be set")
            return self._addtionallyAdoptVP(user_id, mainItemset, itemset, coupon)
  
    def _similarity(self, user_id, itemset):

        if not isinstance(itemset, Itemset):
            itemset = self._itemset[itemset]

        itemset_topic = itemset.topic
        user_topic = self._graph.nodes[user_id]["topic"]

        sim = 0.0
        for i in range(len(itemset_topic)):
            sim += itemset_topic[i]*user_topic[i]

        return sim

    def _mainItemsetVP(self, user_id, itemset):
    
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
        return self._similarity(user_id,itemset)/result.price if not result.empty() else 0
  
    def _adoptMainItemset(self, user_id):
        '''
            從使用者的 desired set 找出CP值最高的商品組合當作主商品並且回傳

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
        if desired_set == None or adopted_set == desired_set:
            return None

        if adopted_set == None:
            adopted_set = set()
            

        max_VP = sys.float_info.min
        maxVP_mainItemset = None

        for length in range(len(desired_set.numbering)):
            for X in combinations(desired_set.numbering, length+1):
                # the combination of iterator returns tuple type
                X = set(X)
                if self._itemset.issubset(adopted_set, X) and adopted_set != X:

                    VP = self._VP_ratio(user_id, X)
                    logging.debug("User {0}, items {1}, VP ratio {2}".format(user_id, self._itemset[X], VP))
                    if VP > max_VP:
                        max_VP = VP
                        maxVP_mainItemset = X

        if maxVP_mainItemset == None or self._itemset[maxVP_mainItemset] == adopted_set:
            logging.debug("User {0} did not adopt any new item.".format(user_id))
            return None
        
        return {"items": self._itemset[maxVP_mainItemset], "VP": max_VP} if maxVP_mainItemset != None else None

    def _addtionallyAdoptVP(self, user_id, mainItemset, itemset, coupon):
        '''
            計算額外購買的VP值
            Args:
                user_id(str): node id
                mainItemset(Itemset): Main itemset of the node
                addtionalItemset(Itemset): 使用者在第二階段考量的商品組合
        '''
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

    def _adoptAddtional(self, user_id, mainItemset):
        '''
            第二購買階段，即考量優惠方案的情況下

            Args:
                mainItemset(Itemset): 第一購買階段決定的商品組合，此組合為考量的商品加上曾購買過的商品
        '''
        

        def parallelAdopt(args):
            result = None
            mainItemset, itemset_instance, coupon = args[0], args[1], args[2]

            if self._itemset.issuperset(itemset_instance, mainItemset):
                # X 必須超過滿額門檻
                diff_adopted = self._itemset.difference(itemset_instance, self._graph.nodes[user_id]["adopted_set"])
                intersection = self._itemset.intersection(diff_adopted, coupon.accItemset)
                
                if intersection != None and intersection.price > coupon.accThreshold:
                    VP = self._addtionallyAdoptVP(user_id, mainItemset, itemset_instance, coupon)
                    logging.debug("User {0}, items {1}, VP_ratio {2}, Coupon {3}".format(user_id, itemset_instance, VP, coupon))
                    result = {"items": itemset_instance, "VP": VP, "coupon": coupon}

            return result

        pool = ThreadPool()
        params = []
        for numbering, itemsetObj in self._itemset:
            for coupon in self._coupons:
                params.append([mainItemset, itemsetObj, coupon])
            
        resultList = pool.map(parallelAdopt, params)
        pool.close()
        maxVP = {"VP": sys.float_info.min}

        for comp_result in resultList:
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
        #                 logging.debug("User {0}, items {1}, VP_ratio {2}, Coupon {3}".format(user_id, itemset_instance, VP, self._coupons[i]))

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
        logging.debug("Adopt Main Itemset")
        mainItemset = self._adoptMainItemset(user_id)

        # 主商品皆為已購買過的商品
        # main itemset should be check whether is empty

        if mainItemset == None or self._itemset.issubset(mainItemset["items"], self._graph.nodes[user_id]["adopted_set"]):
            logging.debug("User {0}'s main itemset is subset equal its adopted set.".format(user_id))
            return None

        trade = dict()
        

        trade["decision_items"] = mainItemset["items"]
        trade["tradeOff_items"] = self._itemset.difference(trade["decision_items"], self._graph.nodes[user_id]["adopted_set"])
        trade["amount"] = trade["tradeOff_items"].price
        trade["coupon"] = None
        trade["VP"] = mainItemset["VP"]
        #logging.info("user {0} choose main itemset {1}.".format(user_id, mainItemset["items"]))
        if self._coupons != None or len(self._coupons) != 0:
            logging.info("Adopt Addtional Itemset")
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
                                    trade["tradeOff_items"])

            self._graph.nodes[user_id]["adopted_records"].append([trade["tradeOff_items"], trade["coupon"], trade["amount"]])
            return trade
        else:
            return None