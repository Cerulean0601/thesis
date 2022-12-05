
from model import DiffusionModel
from coupon import Coupon

class algorithm:
    def __init__(self, model:DiffusionModel):
        self.model = model

    def _generateCoupons(self, price_step:int|float ):
        '''
            Generates a set of all possible coupons.
        '''
        coupons = []
        for threshold in range(min(self.model._itemset.PRICE), price_step):
            for accNumbering, accItemset in self.model._itemset:
                for discount in range(min(self.model._itemset.PRICE), price_step):
                    for disNumbering, disItemset in self.model._itemset:
                        