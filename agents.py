from collections import deque
import numpy as np
from random import sample
from tqdm import tqdm

from configs import model_config, agent_config, environment_config
from environment import RubiksCubeEnvironment
from models import Sequential


class Avg:
  def __init__(self, value=0, n=0):
    self.value = value
    self.n = n

  def update(self, value):
    if value is not None:
      self.value = (self.value * self.n + value) / (self.n + 1 )
      self.n = self.n + 1

  def __str__(self):
    return str(round(self.value, 4))


class DQNAgent:
  def __init__(self, config):
    self.config = config
    self.environment = RubiksCubeEnvironment(environment_config)
    self.exploration_rate = config.max_exploration_rate
    self.model = Sequential(model_config)
    self.memory = deque(maxlen=config.memory_size)

  def remember(self, state, action, reward, next_state):
    self.memory.append((state, action, reward, next_state))

  def policy(self, state):
    if np.random.rand() < self.exploration_rate:
      return np.random.choice(range(self.config.action_size))
    return self.model.act(state)

  def play_episode(self, difficulty, is_eval=False):
    self.environment = RubiksCubeEnvironment(environment_config)
    self.environment.scramble(difficulty)
    state = self.environment.encoded_state()
    is_terminal = False
    counter = 0
    while not is_terminal:
      action = self.model.act(state) if is_eval else self.policy(state)
      reward, next_state, is_terminal = self.environment(action)
      self.remember(state, action, reward, next_state)
      counter += 1
      state = next_state
    return counter, reward > self.environment.config.fail_reward

  def train_on_batch(self):
    if len(self.memory) < self.model.config.batch_size:
      return
    batch = sample(self.memory, self.model.config.batch_size)
    states, actions, rewards, next_states = zip(*batch)
    next_state_values = self.model.session.run(
      fetches=self.model.state_value_prediction,
      feed_dict={self.model.states: next_states}
    )
    values = np.array(rewards) + self.config.gamma * next_state_values
    return self.model.train(states, actions, values)

  def evaluation(self, n=1000):
    avg_length = Avg()
    avg_success = Avg()
    for _ in tqdm(range(n)):
      l, s = agent.play_episode(26, is_eval=True)
      avg_length.update(l)
      avg_success.update(s)
    print("evaluation success ratio: ", avg_success)
    return avg_length, avg_success


if __name__ == "__main__":
  agent = DQNAgent(agent_config)

  for epoch in range(1, 50):
    avg_length = Avg()
    avg_success = Avg()
    avg_loss = Avg()
    for step in range(6000):
      l, s = agent.play_episode(epoch)
      loss = agent.train_on_batch()
      avg_length.update(l)
      avg_success.update(s)
      avg_loss.update(loss)
      if step % 10 == 0:
        print(
          "epoch: {}; step: {}, loss: {}; len: {}, suc: {}".format(
            epoch, step, avg_loss, avg_length, avg_success
          )
        )
      if step % 1000 == 0:
        agent.exploration_rate = max(
          agent_config.min_exploration_rate,
          agent_config.max_exploration_rate * agent.exploration_rate,
        )
        agent.model.save_weights('logdir/e{}_s{}_{}'.format(epoch, step, agent.evaluation()[1]))




