import unittest
from itertools import repeat

from cube.movegen import get_zero_cube
from cube.tensorcube import TensorCube


class TensorCubeTest(unittest.TestCase):
  def test_a_simple_move(self):
    c = TensorCube()
    s, r = c.action(list(repeat(0, 100)), list(repeat(get_zero_cube(), 100)))
    self.assertFalse(s.all())
    s, r = c.action(list(repeat(1, 100)), r)
    self.assertTrue(s.all())
