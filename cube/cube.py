import numpy as np
from .movegen import movegen, get_zero_cube


class Cube:

  def __init__(self):
    self.zerocube = get_zero_cube()
    self.state = self.zerocube
    self.movements = movegen()

  @property
  def solved(self):
    return (self.state == self.zerocube).all()

  def move(self, move, verbose=True):
    self.state = np.matmul(self.movements[move], self.state)
    if verbose:
      return self.state

  def reset(self):
    self.state = self.zerocube
