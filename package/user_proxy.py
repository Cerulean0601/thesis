from .social_graph import SN_Graph
from .itemset import ItemsetFlyweight, Itemset
import sys
from itertools import combinations

class UsersProxy():
    '''
      使用者的購買行為，包含挑選主商品、額外購買、影響成功後的行為
    '''
    def __init__(self, graph: SN_Graph, itemset: ItemsetFlyweight, coupons) -> None:
        self._graph = graph
        self._itemset = itemset
        self._coupons = coupons

    def _VP_ratio(self, user_id, itemset, mainItemset = None, coupon = None):
        if coupon == None:
            return self._mainItemsetVP(user_id, itemset)
        else:
            if mainItemset == None:
                raise ValueError("If coupon isn't None, the mainItemset should be set")
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
                If the result adopted itemset of the user minus the itemset is empty, return None
        '''
        adopted = self._itemset[self._graph.nodes[user_id]["adopted_set"]]
        if not isinstance(itemset, Itemset):
            itemset = self._itemset[itemset]
        if type(itemset) == str:
            print(itemset)
        result = self._itemset.difference(itemset ,adopted)

        return self._similarity(user_id,itemset)/result.price if result != None else None
  
    def _adoptMainItemset(self, user_id):
        '''
            從使用者的 desired set 找出CP值最高的商品組合當作主商品並且回傳

            Args:
                user_id (str): user id
            Returns:
                (Itemset, float): If user's desired_set is empty, or the main itemset and adopted set is equivalance, 
                None will be returned. Otherwise, it will return the main itemset of this user with the value.
        '''

        if user_id not in self._graph:
            raise ValueError("The user id is not found.")

    # If desired set is empty, return None
        desired_items = self._graph.nodes[user_id]["desired_set"]
        adopted_items = self._graph.nodes[user_id]["adopted_set"]
        if desired_items == None:
            return None

        desired_set = self._itemset[desired_items]
        adopted_set = self._itemset[adopted_items]

        if adopted_set == None:
            adopted_set = set()

    # If adopted set and desired set is equivalance, return None 
        if adopted_set == desired_set:
            return None

        max_VP = sys.float_info.min
        maxVP_mainItemset = None

        for length in range(len(desired_items)):
            for c in combinations(desired_items, length+1):
                c = set(c)
                if adopted_set.issubset(c) and not adopted_set == c:

                    VP = self._VP_ratio(user_id, c)
                    if VP > max_VP:
                        max_VP = VP
                        maxVP_mainItemset = c

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
        accThreshold = self._itemset.intersection(mainItemset, coupon.accItemset).price # 已累積的金額
        priceThreshold = coupon.accThreshold
        ratio = min(accThreshold/priceThreshold, 1) # 滿額佔比 \phi

        # 扣除已擁有的商品後，實際交易的商品
        dealItemset = self._itemset.difference(itemset, self._graph.nodes[user_id]["adopted_set"])
        if dealItemset == None:
            raise ZeroDivisionError("Itemset and adopted set of the user should be equivalance.")

        amount = dealItemset.price
        sim = self._similarity(user_id, itemset)

        if dealItemset.price >= coupon.accThreshold:
            disItemset = self._itemset.intersection(dealItemset, coupon.disItemset)
            dealDiscount = min(disItemset.price, coupon.discount)
            amount -= dealDiscount


        return ratio*(sim/amount) if amount != 0 else sys.float_info.max

    def _adoptAddtional(self, user_id, mainItemset):
        '''
            第二購買階段，即考量優惠方案的情況下

            Args:
                mainItemset(Itemset): 第一購買階段決定的商品組合，此組合為考量的商品加上曾購買過的商品
        '''
        max_VP = sys.float_info.min
        maxDict = None # {"items": None, "VP": max_VP, "coupon": None}

        for key, itemset in self._itemset.items():
            # Find the itemset X which is a superset of main itemset
            if itemset.issuperset(mainItemset):
                for i in range(len(self._coupons)):
                    # X 必須超過滿額門檻
                    if self._itemset.intersection(itemset, self._coupons[i].accItemset).price > self._coupons[i].accThreshold:
                        VP = self._addtionallyAdoptVP(user_id, mainItemset, itemset, self._coupons[i])
                        if(VP > max_VP):
                            max_VP = VP
                            maxDict = {"items": itemset, "VP": max_VP, "coupon": self._coupons[i]}

        return maxDict
    
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
        return min(actuallyDis.price, coupon.discount)
    
    def adopt(self, user_id):
        '''
            使用者購買行為，若挑選的主商品皆為已購買過的商品則不會產生任何的購買行為，並且回傳None。
            將實際交易的商品存到使用者的 adopted set，回傳考量的商品組合。

            Return:
                dict: 
                    decision_items 考量的商品組合
                    amount 交易金額，若有搭配優惠方案則已扣除折抵金額。

                None: 未發生購買行為
        '''
        mainItemset = self._adoptMainItemset(user_id)

    # 主商品皆為已購買過的商品
        if mainItemset == None:
            return None

    # main itemset should be check whether is empty
        addtional = self._adoptAddtional(user_id, mainItemset["items"])
        trade = dict()

        if mainItemset["VP"] > addtional["VP"]:
            trade["decision_items"] = mainItemset["items"]
            trade["tradeOff_items"] = self._itemset.difference(trade["decision_items"], self._graph.nodes[user_id]["adopted_set"])
            trade["amount"] = trade["tradeOff_items"].price
        else:
            trade["decision_items"] = addtional["items"]
            trade["tradeOff_items"] = self._itemset.difference(trade["decision_items"], self._graph.nodes[user_id]["adopted_set"]) 
            trade["amount"] = trade["tradeOff_items"].price - self._discount(trade["tradeOff_items"], addtional["items"]["coupon"])
            
        self._graph.nodes[user_id]["adopted_set"] = self._itemset.union(
                                self._graph.nodes[user_id]["adopted_set"],
                                trade["tradeOff_items"])
        return trade