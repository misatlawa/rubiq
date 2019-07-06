import numpy as np
from pycuber import Cube

class RubiksCubeEnvironment:
  sides = 'ULFRBD'
  colors = 'yrgobw'

  def __init__(self, config):
    self.config = config
    self.cube = Cube()
    self.counter = 0
    self.centres = [face[1][1].copy() for face in self.faces()]

  def faces(self):
    return [np.array(self.cube.get_face(d)) for d in self.sides]

  def is_solved(self):
    for face, centre in zip(self.faces(), self.centres):
      if np.any(face != centre):
        return False
    return True

  def encoded_state(self):
    encoded_faces = [f == c for c in self.centres for f in self.faces()]
    return np.stack(encoded_faces).reshape([self.config.state_size]).astype(float)

  def scramble(self, difficulty=26):
    moves = np.random.choice(
      self.config.allowed_moves,
      difficulty,
      replace=True
    )
    self.cube(' '.join(moves))

  def __call__(self, action_id=None):
    if action_id is not None:
      self.cube.perform_step(self.config.allowed_moves[action_id])

    is_terminal = self.is_solved()
    if self.counter < self.config.max_steps and is_terminal:
      reward = self.config.success_reward
    elif self.counter < self.config.max_steps:
      reward = self.config.step_reward
    else:
      reward = self.config.fail_reward

    return reward, self.encoded_state(), is_terminal
