import unittest
import numpy as np
from Furby_p3.utility import *

class TestUtility(unittest.TestCase):

    def test_tscrunch(self):
        self.assertEqual(tscrunch(np.array([2]), 1)[0], 2)
        self.assertEqual(len(tscrunch(np.array([1, 2, 3, 4]), 2)), 2)

