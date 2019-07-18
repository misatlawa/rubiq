import unittest
from itertools import permutations, chain
from .base import *

class AxialMatrixTest(unittest.TestCase):
  def matrix_test(self, matrix, cc=zc, turns=4):
    for i in range(1, turns + 1):
      cc = np.matmul(matrix, cc)
      self.assertEqual(is_solved(cc), i % 4 == 0)
    return cc

  def swing_test(self, cc=zc, matrices=()):
    for i in chain(*permutations(matrices, 2)):
      cc = np.matmul(i, cc)
    self.assertTrue(is_solved(cc))

  def test_a_ro(self):
    self.matrix_test(ro)

  def test_b_gb(self):
    self.matrix_test(gb)

  def test_c_yw(self):
    self.matrix_test(yw)

  def test_d_oturn(self):
    self.matrix_test(oturn)

  def test_e_swing_ro(self):
    self.swing_test(matrices=(ro, ro_inv))

  def test_f_swing_gb(self):
    self.swing_test(matrices=(gb, gb_inv))

  def test_g_swing_yw(self):
    self.swing_test(matrices=(yw, yw_inv))

  def test_h_swing_oturn(self):
    self.swing_test(matrices=(oturn, oturn_inv))


class CubeIntegrityTest(unittest.TestCase):

  def test_a_RUru_sequence(self):
    turns = [oturn, gb, gb_inv, oturn_inv]
    sequence = [0, 1, 0, 2, 3, 1, 3]
    cc = zc
    for i in range(1, 7):
      for turn in sequence:
        cc = np.matmul(turns[turn], cc)
    self.assertEqual(is_solved(cc), i % 6 == 0)


if __name__ == '__main__':
  unittest.main()
