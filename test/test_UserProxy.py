import unittest
from package.itemset import ItemsetFlyweight, Itemset
from package.social_graph import SN_Graph 
from package.coupon import Coupon
from package.user_proxy import UsersProxy

class TestUserProxy(unittest.TestCase):
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
        self._graph = SN_Graph(located=False)
        self._coupons = [Coupon(180, ["0"], 20, ["0","1"]),]
        self._user_proxy = UsersProxy(self._graph, self._itemset, self._coupons, 0)

        return super().setUp()

    def tearDown(self) -> None:
        self._graph.clear()
        return super().tearDown()
        
    def test_calculateVP(self):
        # calculate VP without coupon
        self._graph.add_node("user", adopted_set=None, desired_set=None, topic=[0.2, 0.6])
        ratio = self._user_proxy._VP_ratio("user", self._itemset["1"])
        self.assertAlmostEqual(ratio, 0.001076923076923077)

        self._graph.add_node("with_adopted_set", adopted_set=self._itemset["0"], desired_set=self._itemset["0 1"], topic=[0.2, 0.6])
        ratio = self._user_proxy._VP_ratio("with_adopted_set", self._itemset["0 1"])
        self.assertAlmostEqual(ratio, 0.00230769230769230769230769230769)

        # 超過滿額門檻
        coupon = Coupon(210, ["1"],
                50, ["2"])
        self._graph.add_node("coupon_user", adopted_set=self._itemset["0"], desired_set=self._itemset["0 1"], topic=[0.2, 0.6])
        ratio = self._user_proxy._VP_ratio("coupon_user", self._itemset["0 1 2"], self._itemset["0 1"], coupon)
        self.assertAlmostEqual(ratio, 0.002)

    def test_adoptMainItemset(self):
        # test desired_set is empty
        self._graph.add_node("desired_set_empty", adopted_set=None, desired_set=None, topic=[0.2, 0.6])
        self.assertIsNone(self._user_proxy._adoptMainItemset("desired_set_empty"))

        # test adopted set and deisred set is equivalance
        self._graph.add_node("equivalance", adopted_set=self._itemset["0 1"], desired_set=self._itemset["0 1"], topic=[0.2, 0.6])
        self.assertIsNone(self._user_proxy._adoptMainItemset("equivalance"))

        self._graph.add_node("with_adopted_set", adopted_set=self._itemset["0"], desired_set=self._itemset["0 1"], topic=[0.2, 0.6])
        mainItemset = self._user_proxy._adoptMainItemset("with_adopted_set")
        self.assertEqual(mainItemset["items"], self._itemset["0 1"])

        self._graph.add_node("without_adopted_set", adopted_set=None, desired_set=self._itemset["0 1"], topic=[0.2, 0.6])
        mainItemset = self._user_proxy._adoptMainItemset("without_adopted_set")
        self.assertEqual(mainItemset["items"], self._itemset["0"])
  
    def test_adoptAddtionalItemset(self):
    
        # 滿額占比超過和未超過門檻
        self._graph.add_node("over_threshold", adopted_set=None, desired_set=self._itemset["0 1"], topic=[0.2, 0.6])
        mainItemset = self._user_proxy._adoptMainItemset("over_threshold") # ["0"]
        result = self._user_proxy._adoptAddtional("over_threshold", mainItemset["items"])
        self.assertEqual(result["items"], self._itemset["0 2"])

        #-------------------------------------------------------
        self._graph.add_node("addtionally_adopted", adopted_set=self._itemset["0"], desired_set=self._itemset["0 1 2"], topic=[0.2, 0.8])
        
        origin_topic = self._itemset["0 1 2"]
        self._itemset["0 1 2"].topic = [0.1, 0.9]

        self._coupons.extend([Coupon(80, ["0","1","2"], 310, ["0", "1", "2"]),])
        result = self._user_proxy.adopt("addtionally_adopted")
        self.assertEqual(result["decision_items"], self._itemset["0 1 2"])
        self.assertEqual(result["tradeOff_items"], self._itemset["1 2"])

        self._itemset["0 1 2"].topic = origin_topic
        # 交易金額等於折扣金額
    
class TestAdoptMainItemset(unittest.TestCase):
    def setUp(self) -> None:
        itemset_topic = {
            '0': [0.7, 0.3],
            '1': [0.8, 0.2],
            '2': [0.5, 0.5],
            '0 1': [0, 1],
            '0 2': [0.4, 0.6],
            '1 2': [0.3, 0.7],
            '0 1 2': [0.6, 0.7]
        }
        prices = {"0":200, "1":260, "2":60}

        self._itemset = ItemsetFlyweight(prices = prices, topic = itemset_topic)
        self._graph = SN_Graph(node_topic={0:[0.9, 0.1]},located=False)
        self._user_proxy = UsersProxy(self._graph, self._itemset, coupons=[], threshold=0)

        self._graph.add_node(0)
        self._graph._initAllNodes()
        return super().setUp()
    
    def test_empty_desired_set(self):
        result = self._user_proxy._adoptMainItemset(0)
        self.assertIsNone(result)

    def test_desired_set(self):
        self._graph.nodes[0]["desired_set"] = self._itemset["0 1"]
        result = self._user_proxy._adoptMainItemset(0)
        self.assertEqual(result["items"], self._itemset["0"])
        self.assertAlmostEqual(result["VP"], 0.0033)

    def test_desired_set_equal_adopted_set(self):
        self._graph.nodes[0]["desired_set"] = self._itemset["0 1"]
        self._graph.nodes[0]["adopted_set"] = self._itemset["0 1"]
        result = self._user_proxy._adoptMainItemset(0)
        self.assertIsNone(result)
    
