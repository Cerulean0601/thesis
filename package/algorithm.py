
from itemset import ItemsetFlyweight
from social_graph import SN_Graph
from coupon import Coupon

class Algorithm:
    def __init__(self, graph:SN_Graph, itemset:ItemsetFlyweight):
        self._graph = graph
        self._itemset = itemset

    def greedy(self, price_step:float):
        '''
            Generates a set of all possible coupons.
        '''
        coupons = []
        for threshold in range(min(self.itemset.PRICE), price_step):
            for accNumbering, accItemset in self._itemset:
                for discount in range(price_step):
                    for disNumbering, disItemset in self._itemset:
                        coupons.append(Coupon(threshold, accNumbering, discount, disNumbering))
    
        return coupons