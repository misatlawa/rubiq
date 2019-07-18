import unittest
from cube.cube import Cube


class CubeTest(unittest.TestCase):
  def test_a_solved(self):
    c = Cube()
    self.assertTrue(c.solved)

  def test_b_shuffled(self):
    c = Cube()
    c.shuffle(40)
    self.assertFalse(c.solved)

  def test_c_simple_reverse(self):
    c = Cube()
    self.assertEqual(c.reverse_sequence('RuUr'), 'RuUr')

  def test_d_reverse_solve(self):
    c = Cube()
    seq = c.shuffle(40)
    self.assertFalse(c.solved)
    c.sequencer(c.reverse_sequence(seq))
    self.assertTrue(c.solved)

  def test_e_reset(self):
    c = Cube()
    seq = c.shuffle(40)
    self.assertFalse(c.solved)
    c.reset()
    self.assertTrue(c.solved)

  def test_f_RUru(self):
    c = Cube()
    for i in range(1, 7):
      c.sequencer('RUru')
      self.assertEqual(c.solved, i % 6 == 0)

  def test_g_FRUruf(self):
    c = Cube()
    for i in range(1, 7):
      c.sequencer('FRUruf')
      self.assertEqual(c.solved, i % 6 == 0)
