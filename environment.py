from cube.cube import Cube

class RubiksCubeEnvironment:

  def __init__(self, config):
    self.config = config
    self.cube = Cube()

  def is_solved(self):
    return self.cube.solved

  def encoded_state(self):
    return self.cube.state.reshape([self.config.state_size]).astype(float)

  def scramble(self, difficulty=26):
    self.cube.shuffle(moves=difficulty)

  def reset(self):
    self.cube.reset()

  def __call__(self, action_id=None):
    if action_id is not None:
      self.cube.move(self.config.allowed_moves[action_id])

    is_solved = self.is_solved()
    if is_solved:
      reward = self.config.success_reward
    else:
      reward = self.config.step_reward

    return reward, self.encoded_state(), is_solved
