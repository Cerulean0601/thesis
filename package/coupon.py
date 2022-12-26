from itemset import Itemset

class Coupon():

    def __init__(self, accThreshold:float, accItemset, discount:float, disItemset):
        '''
        It is a sturcture for coupons.

        Arg:
            accThreshold (float): The threshold of discounting.
            accItemset (list|string): Accumlative items that can be redeemed for discounts, if 
                the total price is equal or greater than the threshold.
            discount (float): discount amount
            disItemset (list|string): The items that can be discount.
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

