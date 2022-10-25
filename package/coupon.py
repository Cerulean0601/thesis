from itemset import Itemset

class Coupon():

    def __init__(self, accThreshold, accItemset, discount, disItemset):
        '''
        It is a sturcture for coupons.

        Arg:
            itemHandler (ItemsetFlyweight)
            param (list):
                accThreshold (int): The threshold of discounting.
                accItemset (Itemset): Accumlative items that can be redeemed for discounts, if 
                    the total price is equal or greater than the threshold.
                discount (int): discount amount
                disItemset (Itemset): The items that can be discount.
        '''
        self.accThreshold = accThreshold
        self.accItemset = accItemset
        self.discount = discount
        self.disItemset = disItemset

    def __str__(self):
        return "(" + str(self.accThreshold) + \
        ", {" + str(self.accItemset) + "}" +\
        ", " + str(self.discount) + \
        ", {" + str(self.disItemset) + "})"

