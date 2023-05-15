from package.itemset import Itemset

class Coupon():

    def __init__(self, accThreshold:float, accItemset:Itemset, discount:float, disItemset:Itemset):
        '''
        It is a sturcture for coupons.

        Arg:
            accThreshold (float): The threshold of discounting.
            accItemset (Itemset): Accumlative items that can be redeemed for discounts, if 
                the total price is equal or greater than the threshold.
            discount (float): discount amount
            disItemset (Itemset): The items that can be discount.
        '''
        self.accThreshold = accThreshold
        self.accItemset = accItemset
        self.discount = discount
        self.disItemset = disItemset

    def __str__(self):
        return str(self.accThreshold) + \
        "," + str(self.accItemset) + \
        "," + str(self.discount) + \
        "," + str(self.disItemset)

