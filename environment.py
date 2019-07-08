import numpy as np
from cube.cube import Cube

class RubiksCubeEnvironment:

  def __init__(self, config):
    self.config = config
    self.cube = Cube()
    self.counter = 0

  def is_solved(self):
    return self.cube.solved

  def encoded_state(self):
    return self.cube.state.reshape([self.config.state_size]).astype(float)

  def scramble(self, difficulty=26):
    self.cube.shuffle(moves=difficulty)

  def __call__(self, action_id=None):
    if action_id is not None:
      self.cube.move(self.config.allowed_moves[action_id])

    is_terminal = self.is_solved()
    if self.counter < self.config.max_steps and is_terminal:
      reward = self.config.success_reward
    elif self.counter < self.config.max_steps:
      reward = self.config.step_reward
    else:
      reward = self.config.fail_reward
      is_terminal = True
    self.counter += 1
    return reward, self.encoded_state(), is_terminal
