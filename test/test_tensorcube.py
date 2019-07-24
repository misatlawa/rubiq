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

  def test_b_few_moves(self):
    c = TensorCube()
    r = list(repeat(get_zero_cube(), 100))

    for move in filter(lambda x: x % 2 == 0, range(12)):
      s, r = c.action(list(repeat(move, 100)), r)
    self.assertFalse(s.all())

    for move in filter(lambda x: x % 2 != 0, range(12)):
      s, r = c.action(list(repeat(move, 100)), r)
    self.assertTrue(s.all())

  def test_c_RUru(self):
    c = TensorCube()
    sequence = [0, 8, 1, 9]
    r = list(repeat(get_zero_cube(), 100))

    for i in range(1, 7):
      for move in sequence:
        s, r = c.action(list(repeat(move, 100)), r)
      self.assertEqual(s.all(), i % 6 == 0)

