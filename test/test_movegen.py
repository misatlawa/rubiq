import unittest
from cube.movegen import movegen

from .base import *

moves = movegen()


class MovegenTest(unittest.TestCase):

  def cycle_test(self, move, cc=zc, turns=4):
    for i in range(1, turns + 1):
      cc = np.matmul(move, cc)
      self.assertEqual(is_solved(cc), i % 4 == 0)

  def test_a_U(self):
    self.cycle_test(moves['U'])

  def test_b_u(self):
    self.cycle_test(moves['u'])

  def test_c_D(self):
    self.cycle_test(moves['D'])

  def test_d_d(self):
    self.cycle_test(moves['d'])

  def test_e_F(self):
    self.cycle_test(moves['F'])

  def test_f_f(self):
    self.cycle_test(moves['f'])

  def test_g_B(self):
    self.cycle_test(moves['B'])

  def test_h_b(self):
    self.cycle_test(moves['b'])

  def test_i_L(self):
    self.cycle_test(moves['L'])

  def test_j_l(self):
    self.cycle_test(moves['l'])

  def test_k_R(self):
    self.cycle_test(moves['R'])

  def test_l_r(self):
    self.cycle_test(moves['r'])

  def test_m_RUru(self):
    cc = zc
    sequence = ['R', 'U', 'r', 'u']
    for i in range(1, 7):
      for move in sequence:
        cc = np.matmul(moves[move], cc)
      self.assertEqual(is_solved(cc), i % 6 == 0)
