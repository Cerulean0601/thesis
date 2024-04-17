import unittest
from package.itemset import ItemsetFlyweight, Itemset
from package.social_graph import SN_Graph 
from package.coupon import Coupon
from package.user_proxy import UsersProxy
from package.tag import Tagger

class TestUserProxy(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestUserProxy, self).__init__(*args, **kwargs)

    def setUp(self) -> None:
        topic = {
            '0': [0.7, 0.3],
            '1': [0.8, 0.2],
            '2': [0.5, 0.5],
            '0 1': [0, 1],
            '0 2': [0.4, 0.6],
            '1 2': [0.3, 0.7],
            '0 1 2': [0.6, 0.7]
        }
        prices = {"0":200, "1":260, "2":60}
        self._itemset = ItemsetFlyweight(prices = prices, topic = topic)
        self._graph = SN_Graph()
        self._coupons = [Coupon(180, ["0"], 20, ["0","1"]),]
        self._user_proxy = UsersProxy(self._graph, self._itemset, self._coupons, 0)

        return super().setUp()

    def tearDown(self) -> None:
        self._graph.clear()
        return super().tearDown()
        
    def test_countNonActiveNode(self):
        pass