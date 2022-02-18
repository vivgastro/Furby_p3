import unittest
import numpy as np

from Furby_p3.Signal import *

class TestSignal(unittest.TestCase):

    def test_bw(self):
        tel = Telescope(800, 700, 100, 1, "FAKE")
        self.assertEqual(tel.bw, 100)
        self.assertEqual(tel.chw, 1)
