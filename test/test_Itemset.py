import unittest
from itemset import ItemsetFlyweight 

class TestItemset(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestItemset, self).__init__(*args, **kwargs)
        topic = {
            '0': [0.82, 0.19],
            '1': [0.63, 0.37],
            '2': [0.5, 0.5]
        }
        price = [60,260,70]
        self._itemset = ItemsetFlyweight(price = price, topic = topic)
  
    def test_union(self):
        # NOTICE! the aggregation is not determinated

        self.assertEqual(self._itemset.union({0},{1}).price, 320)
        self.assertEqual(self._itemset.union({0,1},{1,2}).price, 390)

    def test_intersection(self):
        self.assertIsNone(self._itemset.intersection({0},{1}))
        self.assertEqual(self._itemset.intersection({0,1},{1,2}).price, 260)

    def test_difference(self):
        self.assertIsNone(self._itemset.difference({1},{1,2,3}))
        self.assertIsNone(self._itemset.difference({1,2},{1,2,3}))
        self.assertEqual(self._itemset.difference({0,1},{1,2}).price, 60)
