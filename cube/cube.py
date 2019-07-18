import numpy as np
from .movegen import movegen, get_zero_cube
from random import choice

class Cube:
  def __init__(self):
    self.zerocube = get_zero_cube()
    self.movements = movegen()
    self.state = self.zerocube

  @classmethod
  def reverse_sequence(cls, sequence):
    return sequence[::-1].swapcase()

  @property
  def solved(self):
    return (self.state == self.zerocube).all()

  def move(self, move, verbose=True):
    self.state = np.matmul(self.movements[move], self.state)
    if verbose:
      return self.state

  def reset(self):
    self.state = self.zerocube

  def sequencer(self, sequence, verbose=True):
    for move in sequence:
      self.move(move, verbose=False)
    if verbose:
      return self.state

  def shuffle(self, moves):
    sequence = ''.join(choice(list(self.movements.keys())) for _ in range(moves))
    self.sequencer(sequence)
    return sequence
