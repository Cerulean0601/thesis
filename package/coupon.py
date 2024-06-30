from package.itemset import Itemset
import heapq

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
    
    def __lt__(self, other):
        if self.discount == other.discount:
            return len(self.accItemset) < len(other.accItemset)
        return self.discount < other.discount
    
    def __str__(self):
        return str(self.accThreshold) + \
        "," + str(self.accItemset) + \
        "," + str(self.discount) + \
        "," + str(self.disItemset)

class CouponRevenueMaxHeap:
    def __init__(self, tuples):
        # tuple = (revenue, coupon), revenue必須在第一個才能做排序
        self.heap = list(tuples)

    def push(self, coupon, revenue):
        # Push the negative value to simulate a max heap
        heapq.heappush(self.heap, (-revenue, coupon))

    def pop(self):
        # Pop the negative value and return the original value
        revenue, coupon = heapq.heappop(self.heap)
        return -revenue, coupon

    def peek(self):
        # Return the top value without popping it
        revenue, coupon = self.heap[0]
        return -revenue, coupon

    def clear(self):
        self.heap = []
        
    def __len__(self):
        return len(self.heap)

    def __str__(self):
        return str([(-revenue, str(coupon)) for revenue,coupon in self.heap])