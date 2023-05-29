import unittest
from package.itemset import ItemsetFlyweight, ItemRelation,Itemset
import pandas as pd

class TestItemsetFlyweight(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestItemsetFlyweight, self).__init__(*args, **kwargs)
        
    def setUp(self) -> None:
        topic = {
            '0': [0.82, 0.19],
            '1': [0.63, 0.37],
            '2': [0.5, 0.5]
        }
        prices = {'0':60, '1':260, '2':70}
        relation_dict = pd.DataFrame.from_dict({
            '0':{
                '1':5,
                '2':-2
            },
            '1':{
                '0':2,
                '2':-2,
            },
            '2':{
                '0': -1,
                '1':-5,
            }
            }, orient="index")
            
        relation = ItemRelation(relation_dict)
        self._itemset = ItemsetFlyweight(prices, topic, relation)
        return super().setUp()

    def assertListAlomstEqual(self, list1: list, list2: list, place=7) -> None:
        if len(list1) != len(list2):
            raise ValueError("The size of both of the lists must be equal.")

        for i in range(len(list1)):
            diff = abs(list1[i] - list2[i])
            if round(diff, place) != 0:
                raise AssertionError("{0} != {1} at index {2}".format(list1[i], list2[i], i))
                return
        return

    def test_union(self):
        # NOTICE! the aggregation is not determinated

        self.assertEqual(self._itemset.union(["0"],["1"]).price, 320)
        self.assertEqual(self._itemset.union(["0","1"],["1","2"]).price, 390)

    def test_intersection(self):
        self.assertIsNone(self._itemset.intersection(["0"],["1"]))
        self.assertEqual(self._itemset.intersection(["0","1"],["1","2"]).price, 260)

    def test_difference(self):
        self.assertIsNone(self._itemset.difference(["1"],["1","2","3"]))
        self.assertIsNone(self._itemset.difference(["1","2"],["1","2","3"]))
        self.assertEqual(self._itemset.difference(["0","1"],["1","2"]).price, 60)
        
    def test_aggregate(self):
        self.assertListAlomstEqual(self._itemset["0 1 2"].topic, [0.65730239, 0.346236422], 4)
        self.assertListAlomstEqual(self._itemset["0 1 2"].topic, self._itemset["1 2 0"].topic, 4)
        