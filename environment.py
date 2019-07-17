import numpy as np
from cube.cube import Cube

class RubiksCubeEnvironment:

  def __init__(self, config):
    self.state_size = config.state_size
    self.allowed_moves = config.allowed_moves
    self.step_reward = config.step_reward
    self.success_reward = config.success_reward
    self.fail_reward = config.fail_reward
    self.max_steps = config.max_steps

    self.cube = Cube()
    self.counter = 0

  def is_solved(self):
    return self.cube.solved

  def encoded_state(self):
    return self.cube.state.reshape([self.state_size]).astype(float)

  def scramble(self, difficulty=26):
    self.cube.shuffle(moves=difficulty)

  def __call__(self, action_id=None):
    if action_id is not None:
      self.cube.move(self.allowed_moves[action_id])

    is_terminal = self.is_solved()
    if self.counter < self.max_steps and is_terminal:
      reward = self.success_reward
    elif self.counter < self.max_steps:
      reward = self.step_reward
    else:
      reward = self.fail_reward
      is_terminal = True
    self.counter += 1
    return reward, self.encoded_state(), is_terminal
